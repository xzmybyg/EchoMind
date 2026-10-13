import io
import json
from unittest.mock import Mock

import numpy as np
import pytest

from echomind.asr_backends import create_transcriber, MODEL_FILES, experimental_model_path
from echomind.qwen_asr import QwenASRError, QwenTranscriber, runtime_python
from echomind.vad import EnergyVAD


def transcriber(text="解释一下React Hooks", **options):
    result = QwenTranscriber(model_size="unused", vad=EnergyVAD(), **options)
    result._load = Mock()
    result._process = Mock(stdin=io.StringIO())
    result._receive = Mock(return_value={"text": text})
    return result


def test_qwen_disables_prompts_context_and_previews():
    model = transcriber(hotwords="React, Vue", context_window_ms=15000, retry_low_confidence=True)
    assert model.hotwords is None and model.context_window_ms == 0
    assert not model.retry_low_confidence
    assert model.partial_interval_ms > model.max_utterance_ms
    voice = np.full(16000, 800, dtype="<i2").tobytes()
    assert model.push_pcm(voice) == []
    events = model.push_pcm(bytes(660 * 32))
    assert [(e.kind, e.text) for e in events] == [("final", "解释一下React Hooks")]
    assert set(json.loads(model._process.stdin.getvalue())) == {"pcm"}


def test_qwen_silence_never_decodes_but_warmup_does():
    model = transcriber("嗯")
    assert model.push_pcm(bytes(32000 * 10)) == []
    assert model.flush() == []
    assert model._transcribe(bytes(32000))[0] == ""
    model._load.assert_not_called()
    model.warm_up()
    model._load.assert_called_once()


def test_qwen_waits_for_unfinished_request_without_previous_turn_context():
    model = transcriber("请解释一下")
    voice = np.full(14400, 800, dtype="<i2").tobytes()
    first = model.push_pcm(voice + bytes(660 * 32))
    assert first and all(event.kind == "partial" for event in first)
    model._receive.return_value = {"text": "请解释一下React Hooks"}
    events = model.push_pcm(voice + bytes(660 * 32))
    assert len(events) == 1 and events[0].kind == "final"
    assert "React Hooks" in events[0].text


@pytest.mark.parametrize("text,truncated", [("React, Vue, " * 6, False), ("一段截断结果", True)])
def test_qwen_rejections_do_not_become_captions(text, truncated):
    model = transcriber(text)
    model._receive.return_value["truncated"] = truncated
    voice = np.full(16000, 800, dtype="<i2").tobytes()
    events = model.push_pcm(voice + bytes(660 * 32))
    assert len(events) == 1 and events[0].kind == "rejected"
    model._receive.return_value = {"text": "解释一下React Hooks"}
    assert model.push_pcm(voice + bytes(660 * 32))[0].kind == "final"


def test_qwen_failure_timeout_and_cancellation_release_worker():
    for response in (None, {"error": True}):
        model = QwenTranscriber(vad=EnergyVAD())
        model.close = Mock()
        model._responses.put(response)
        with pytest.raises(QwenASRError, match="异常"):
            model._receive(1)
        model.close.assert_called_once()
    model = QwenTranscriber(vad=EnergyVAD())
    model.close = Mock()
    with pytest.raises(QwenASRError, match="超时"):
        model._receive(0)
    model._stop.set()
    assert model._receive(1) is None


def test_close_releases_process_and_pipes_once():
    model = QwenTranscriber(vad=EnergyVAD())
    process = model._process = Mock()
    process.poll.return_value = None
    model.close()
    model.close()
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    process.stdin.close.assert_called_once()
    process.stdout.close.assert_called_once()


def test_factory_reuses_session_owned_worker_and_finds_local_assets(tmp_path, monkeypatch):
    monkeypatch.setattr("echomind.components.cache_root", lambda: tmp_path / "cache")
    marker = tmp_path / "services/transcription/pyproject.toml"
    marker.parent.mkdir(parents=True)
    marker.touch()
    folder = tmp_path / ".models/qwen3-asr-1.7b"
    folder.mkdir(parents=True)
    for name in MODEL_FILES["qwen3-asr-1.7b"]:
        (folder / name).write_bytes(b"fixture")
    assert experimental_model_path(tmp_path / "dist/new/EchoMind", "qwen3-asr-1.7b") == folder
    owned = Mock()
    assert create_transcriber(engine="qwen3-asr-1.7b", model_size=folder, local_transcriber=owned) is owned
    python = tmp_path / ".runtime/qwen17-eval/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture")
    from echomind.qwen_components import runtime_ready
    for name in ("torch/__init__.py", "torch/lib/torch_cuda.dll", "transformers/__init__.py",
                 "librosa/__init__.py", "numpy/__init__.py", "soundfile.py"):
        path = python.parent.parent / "Lib/site-packages" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assert runtime_ready(python.parent.parent)
    # Application root is searched first, so isolate it from the real deployment.
    monkeypatch.setattr("echomind.qwen_asr.__file__", str(tmp_path / "services/transcription/echomind/qwen_asr.py"))
    assert runtime_python(folder) == python


def test_desktop_qwen_uses_gpu_and_closes_even_if_capture_fails(monkeypatch, tmp_path):
    from echomind.desktop_session import DesktopSession, SessionConfig
    monkeypatch.setattr("echomind.desktop_session.validate_model", lambda *_: None)
    monkeypatch.setattr("echomind.desktop_session.runtime_python", lambda *_: tmp_path)
    factory = Mock()
    monkeypatch.setattr("echomind.desktop_session.QwenTranscriber", factory)
    def fail(*args, **kwargs):
        assert args[1] == "cuda"
        assert kwargs["asr_options"]["local_transcriber"] is factory.return_value
        raise RuntimeError("capture failure")
    monkeypatch.setattr("echomind.desktop_session.run_microphone", fail)
    config = SessionConfig(tmp_path, tmp_path, tmp_path, microphone=True, asr_engine="qwen3-asr-1.7b")
    session = DesktopSession(config, lambda *_: None)
    session.start()
    assert session.wait(3)
    factory.return_value.close.assert_called_once()
