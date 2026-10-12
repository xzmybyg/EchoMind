from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from echomind.asr_backends import (MODEL_FILES, ParaformerTranscriber, SenseVoiceTranscriber,
                                  create_transcriber, experimental_model_path, validate_model)


def test_default_factory_preserves_whisper(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("echomind.asr.Transcriber", factory)
    assert create_transcriber(model_size="existing", device="cuda") is factory.return_value
    factory.assert_called_once_with(model_size="existing", device="cuda")


@pytest.mark.parametrize("engine", ["sensevoice", "paraformer"])
def test_experimental_factory_requires_matching_model_files(tmp_path, engine):
    with pytest.raises(ValueError, match="不完整"):
        create_transcriber(engine=engine, model_size=str(tmp_path))
    for name in MODEL_FILES[engine]:
        (tmp_path / name).write_bytes(b"fixture")
    result = create_transcriber(engine=engine, model_size=str(tmp_path), context_window_ms=15000)
    assert result.context_window_ms == (0 if engine == "paraformer" else 15000)
    with pytest.raises(ValueError, match="无效"):
        validate_model(tmp_path, "unknown")


def test_model_resolution_reuses_project_then_portable_and_ignores_empty_files(tmp_path, monkeypatch):
    monkeypatch.setattr("echomind.components.cache_root", lambda: tmp_path / "cache")
    marker = tmp_path / "services/transcription/pyproject.toml"
    marker.parent.mkdir(parents=True)
    marker.touch()
    root = tmp_path / "dist/test/EchoMind"
    project_model = tmp_path / ".models/sensevoice"
    portable = root / "models/sensevoice"
    for folder in (project_model, portable):
        folder.mkdir(parents=True)
        for name in MODEL_FILES["sensevoice"]:
            (folder / name).write_bytes(b"fixture" if folder == project_model else b"")
    assert experimental_model_path(root, "sensevoice") == project_model
    for name in MODEL_FILES["sensevoice"]:
        (portable / name).write_bytes(b"fixture")
    assert experimental_model_path(root, "sensevoice") == portable
    assert experimental_model_path(root, "paraformer") == tmp_path / "cache/models/paraformer"


def test_sensevoice_filters_already_committed_tokens_and_normalizes_chinese():
    stream = Mock()
    stream.result = SimpleNamespace(text="旧問題。新問題。", tokens=["旧問題。", "新問題。"], timestamps=[0.1, 0.7])
    decoder = Mock()
    decoder.create_stream.return_value = stream
    transcriber = SenseVoiceTranscriber(model_size="unused", context_window_ms=15000)
    transcriber._decoder = decoder
    pcm = np.full(16000, 800, dtype="<i2").tobytes()
    text, _ = transcriber._transcribe(pcm, final=True, skip_before_ms=600)
    assert text == "新问题。"
    assert transcriber._last_word_end_ms == 760
    np.testing.assert_allclose(stream.accept_waveform.call_args.args[1], 800 / 32768)


def test_sensevoice_rejects_repetition_before_timestamp_filter():
    stream = Mock()
    stream.result = SimpleNamespace(text="Hooks, Hooks, Hooks, Hooks", tokens=["Hooks"], timestamps=[0.7])
    transcriber = SenseVoiceTranscriber(model_size="unused", context_window_ms=15000)
    transcriber._decoder = Mock()
    transcriber._decoder.create_stream.return_value = stream
    transcriber._transcribe(np.full(16000, 800, dtype="<i2").tobytes(), final=True, skip_before_ms=600)
    assert transcriber._last_rejection
    assert transcriber._last_suspect


@pytest.mark.parametrize("backend", [SenseVoiceTranscriber, ParaformerTranscriber])
def test_digital_silence_does_not_load_decoder(backend):
    transcriber = backend(model_size="unused")
    transcriber._load = Mock(side_effect=AssertionError("silence must not invoke ASR"))
    assert transcriber._transcribe(bytes(32000), final=True)[0] == ""


def test_paraformer_feeds_incremental_samples_once_and_resets_after_final():
    decoder = Mock()
    decoder.is_ready.return_value = False
    decoder.get_result.return_value = "讲解一下React"
    first, second = Mock(), Mock()
    decoder.create_stream.side_effect = [first, second]
    transcriber = ParaformerTranscriber(model_size="unused")
    transcriber._decoder = decoder
    pcm = np.full(3200, 800, dtype="<i2").tobytes()
    transcriber._transcribe(pcm)
    transcriber._transcribe(pcm * 2)
    transcriber._transcribe(pcm * 3, final=True)
    assert [len(call.args[1]) for call in first.accept_waveform.call_args_list] == [3200, 3200, 3200, 8000]
    first.input_finished.assert_called_once()
    assert transcriber._stream is None and transcriber._fed_samples == 0
    transcriber._transcribe(pcm)
    assert len(second.accept_waveform.call_args.args[1]) == 3200


def test_paraformer_anomaly_uses_shared_rejected_event_contract():
    transcriber = ParaformerTranscriber(model_size="unused")
    decoder = Mock()
    decoder.is_ready.return_value = False
    decoder.get_result.return_value = "Vue, JavaScript, Vue, JavaScript, Vue, JavaScript"
    transcriber._decoder = decoder
    transcriber._audio.extend(np.full(3200, 800, dtype="<i2").tobytes())
    assert transcriber._event("partial", 200) is None
    event = transcriber._event("final", 800)
    assert event.kind == "rejected" and event.rejection_reason


def test_paraformer_rejected_preview_resets_decoder_for_next_turn():
    transcriber = ParaformerTranscriber(model_size="unused")
    decoder = Mock()
    decoder.is_ready.return_value = False
    decoder.get_result.side_effect = ["React, Vue, " * 6, "解释一下Hooks"]
    transcriber._decoder = decoder
    voice = np.full(14400, 800, dtype="<i2").tobytes()
    assert transcriber.push_pcm(voice) == []
    assert transcriber._stream is not None
    assert transcriber.push_pcm(bytes(600 * 32))[0].kind == "rejected"
    assert transcriber._stream is None and transcriber._fed_samples == 0
    transcriber.partial_interval_ms = 30000
    events = transcriber.push_pcm(voice[:300 * 32] + bytes(600 * 32))
    assert [(e.kind, e.text) for e in events] == [("final", "解释一下Hooks")]
    assert decoder.create_stream.call_count == 2
