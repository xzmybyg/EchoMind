"""Local experimental ONNX engines sharing the existing PCM/event contract."""
from pathlib import Path
from time import perf_counter

import numpy as np

from .asr import Transcriber, SAMPLE_RATE, _decode_anomaly
from .text import normalize_transcript


ENGINE_LABELS = {"whisper": "Whisper（当前）", "sensevoice": "SenseVoice（实验 / CPU）",
                 "paraformer": "Paraformer（实验 / CPU）", "qwen3-asr-1.7b": "Qwen3-ASR-1.7B（本地 GPU）",
                 "tencent": "腾讯云（在线 / 上传音频）"}
MODEL_FILES = {"whisper": ("config.json", "model.bin", "tokenizer.json", "vocabulary.json"),
               "sensevoice": ("model.int8.onnx", "tokens.txt"),
               "paraformer": ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"),
               "qwen3-asr-1.7b": ("config.json", "generation_config.json", "model.safetensors",
                                  "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}


def validate_model(folder, engine):
    if engine not in MODEL_FILES:
        raise ValueError("识别引擎无效")
    if not all((Path(folder) / name).is_file() for name in MODEL_FILES[engine]):
        raise ValueError("所选本地识别模型不完整，请选择与识别引擎对应的模型目录。")


def experimental_model_path(app_root, engine):
    """Reuse portable, cached, or nearby development models; never download."""
    from .components import cache_root
    if engine not in ("sensevoice", "paraformer", "qwen3-asr-1.7b"):
        raise ValueError("实验识别引擎无效")
    root = Path(app_root).resolve()
    fallback = cache_root() / "models" / engine
    candidates = [root / "models" / engine, fallback]
    project = next((path for path in (root, *root.parents[:4])
                    if (path / "services/transcription/pyproject.toml").is_file()), None)
    if project is not None:
        candidates.append(project / ".models" / engine)
    for folder in candidates:
        if all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in MODEL_FILES[engine]):
            return folder
    return fallback


def create_transcriber(*, engine="whisper", **options):
    if engine == "tencent":
        return options["cloud_transcriber"]
    if engine == "whisper":
        from . import asr
        return asr.Transcriber(**options)
    validate_model(options["model_size"], engine)
    if engine == "qwen3-asr-1.7b":
        from .qwen_asr import QwenTranscriber
        return options.pop("local_transcriber", None) or QwenTranscriber(**options)
    if engine == "sensevoice":
        return SenseVoiceTranscriber(**options)
    return ParaformerTranscriber(**options)


class SenseVoiceTranscriber(Transcriber):
    def __init__(self, **options):
        super().__init__(**options)
        self._decoder = None

    def _load(self):
        if self._decoder is None:
            import sherpa_onnx
            folder = Path(self.model_size)
            self._decoder = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(folder / "model.int8.onnx"), tokens=str(folder / "tokens.txt"),
                num_threads=4, provider="cpu", language="auto", use_itn=True)

    def _transcribe(self, pcm, *, final=False, warmup=False, skip_before_ms=None):
        started = perf_counter()
        self._last_suspect = False
        self._last_original = self._last_rejection = ""
        self._last_word_end_ms = 0.0
        if not warmup and not any(pcm):
            return "", (perf_counter() - started) * 1000
        self._load()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        stream = self._decoder.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._decoder.decode_stream(stream)
        result = stream.result
        text = result.text
        self._last_rejection = _decode_anomaly(text)
        if skip_before_ms is not None:
            # SenseVoice exposes token start times, not Whisper word intervals.
            selected = [(token, timestamp) for token, timestamp in zip(result.tokens, result.timestamps)
                        if 0 <= timestamp * 1000 <= len(pcm) / 32
                        and timestamp * 1000 + 60 >= skip_before_ms]
            text = "".join(token for token, _ in selected)
            if selected:
                self._last_word_end_ms = min(selected[-1][1] * 1000 + 60, len(pcm) / 32)
        text = normalize_transcript(text)
        self._last_rejection = self._last_rejection or _decode_anomaly(text)
        self._last_suspect = bool(self._last_rejection)
        return text, (perf_counter() - started) * 1000


class ParaformerTranscriber(Transcriber):
    def __init__(self, **options):
        # This model carries its own streaming cache and has no word timestamps.
        options["context_window_ms"] = 0
        super().__init__(**options)
        self._decoder = None
        self._stream = None
        self._fed_samples = 0

    def _finish(self, timestamp_ms, *, carry=False):
        events = super()._finish(timestamp_ms, carry=carry)
        if any(event.kind == "rejected" for event in events):
            self._stream = None
            self._fed_samples = 0
        return events

    def _load(self):
        if self._decoder is None:
            import sherpa_onnx
            folder = Path(self.model_size)
            self._decoder = sherpa_onnx.OnlineRecognizer.from_paraformer(
                encoder=str(folder / "encoder.int8.onnx"), decoder=str(folder / "decoder.int8.onnx"),
                tokens=str(folder / "tokens.txt"), num_threads=4, provider="cpu")

    def _transcribe(self, pcm, *, final=False, warmup=False, skip_before_ms=None):
        started = perf_counter()
        self._last_suspect = False
        self._last_original = self._last_rejection = ""
        self._last_word_end_ms = 0.0
        if not warmup and not any(pcm):
            return "", (perf_counter() - started) * 1000
        self._load()
        if self._stream is None:
            self._stream = self._decoder.create_stream()
            self._fed_samples = 0
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self._stream.accept_waveform(SAMPLE_RATE, samples[self._fed_samples:])
        self._fed_samples = len(samples)
        if final or warmup:
            self._stream.accept_waveform(SAMPLE_RATE, np.zeros(8000, dtype=np.float32))
            self._stream.input_finished()
        while self._decoder.is_ready(self._stream):
            self._decoder.decode_stream(self._stream)
        text = normalize_transcript(self._decoder.get_result(self._stream))
        self._last_rejection = _decode_anomaly(text)
        self._last_suspect = bool(self._last_rejection)
        if final or warmup:
            self._stream = None
            self._fed_samples = 0
        return text, (perf_counter() - started) * 1000
