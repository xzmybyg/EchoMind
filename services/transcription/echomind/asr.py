"""Synchronous, incremental Chinese transcription from 16 kHz mono PCM.

``push_pcm`` accepts arbitrary sized little-endian int16 chunks. It can block
while Whisper runs, so callers capturing audio should feed it from a worker.
``timestamp_ms`` is the capture time of the chunk's end; if omitted, events use
the audio stream's elapsed time. ``latency_ms`` measures ASR processing time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

from .vad import EnergyVAD


SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2


@dataclass(frozen=True)
class TranscriptEvent:
    kind: Literal["partial", "final"]
    text: str
    timestamp_ms: int
    latency_ms: float


def _join_without_overlap(previous: str, current: str) -> str:
    """Remove a repeated prefix caused by carrying audio between long windows."""
    if not previous:
        return current
    for length in range(min(len(previous), len(current)), 1, -1):
        if previous.endswith(current[:length]):
            return current[length:]
    return current


class Transcriber:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        *,
        recognizer: Callable[[bytes], str] | None = None,
        vad: EnergyVAD | None = None,
        partial_interval_ms: int = 700,
        min_speech_ms: int = 240,
        end_silence_ms: int = 600,
        max_utterance_ms: int = 12_000,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._recognizer = recognizer
        self._model = None
        self.vad = vad or EnergyVAD()
        self.partial_interval_ms = partial_interval_ms
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.max_utterance_ms = max_utterance_ms

        self._pending = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=7)  # 210 ms
        self._audio = bytearray()
        self._active = False
        self._voice_ms = 0
        self._silence_ms = 0
        self._since_partial_ms = 0
        self._last_partial = ""
        self._previous_final = ""
        self._carried_overlap = False
        self._samples_seen = 0
        self._last_capture_timestamp_ms: int | None = None

    def warm_up(self) -> None:
        """Load the model and complete its first inference before capture starts."""
        if self._recognizer is None:
            self._transcribe(bytes(SAMPLE_RATE * 2))

    def _transcribe(self, pcm: bytes) -> tuple[str, float]:
        started = perf_counter()
        if self._recognizer is not None:
            result = self._recognizer(pcm)
        else:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
            import numpy as np

            samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                samples,
                language="zh",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            result = "".join(segment.text for segment in segments)
        return result.strip(), (perf_counter() - started) * 1000

    def _event(self, kind: Literal["partial", "final"], timestamp_ms: int) -> TranscriptEvent | None:
        text, latency_ms = self._transcribe(bytes(self._audio))
        if self._carried_overlap:
            text = _join_without_overlap(self._previous_final, text)
        if not text or (kind == "partial" and text == self._last_partial):
            return None
        if kind == "partial":
            self._last_partial = text
        else:
            self._previous_final = (self._previous_final + text)[-200:]
        return TranscriptEvent(kind, text, timestamp_ms, latency_ms)

    def _finish(self, timestamp_ms: int, *, carry: bool = False) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        if self._voice_ms >= self.min_speech_ms:
            event = self._event("final", timestamp_ms)
            if event:
                events.append(event)
        tail = bytes(self._audio[-SAMPLE_RATE:]) if carry else b""  # 500 ms
        self._audio = bytearray(tail)
        self._active = carry
        self._voice_ms = 0
        self._silence_ms = 0
        self._since_partial_ms = 0
        self._last_partial = ""
        self._carried_overlap = carry
        if not carry:
            self._preroll.clear()
        return events

    def push_pcm(self, data: bytes, timestamp_ms: int | None = None) -> list[TranscriptEvent]:
        if len(data) % 2:
            raise ValueError("PCM chunks must contain complete int16 samples")
        if timestamp_ms is not None:
            self._last_capture_timestamp_ms = timestamp_ms
        self._pending.extend(data)
        events: list[TranscriptEvent] = []
        while len(self._pending) >= FRAME_BYTES:
            frame = bytes(self._pending[:FRAME_BYTES])
            del self._pending[:FRAME_BYTES]
            self._samples_seen += FRAME_BYTES // 2
            frame_time = self._samples_seen * 1000 // SAMPLE_RATE
            if timestamp_ms is not None:
                frame_time = timestamp_ms - len(self._pending) * 1000 // (SAMPLE_RATE * 2)

            speech = self.vad.is_speech(frame)
            if not self._active:
                self._preroll.append(frame)
                if speech:
                    self._active = True
                    self._audio.extend(b"".join(self._preroll))
                    self._preroll.clear()
                    self._voice_ms = FRAME_MS
                continue

            self._audio.extend(frame)
            self._since_partial_ms += FRAME_MS
            if speech:
                self._voice_ms += FRAME_MS
                self._silence_ms = 0
            else:
                self._silence_ms += FRAME_MS

            if self._silence_ms >= self.end_silence_ms:
                events.extend(self._finish(frame_time))
            elif len(self._audio) >= self.max_utterance_ms * SAMPLE_RATE * 2 // 1000:
                events.extend(self._finish(frame_time, carry=True))
            elif (
                self._voice_ms >= self.min_speech_ms
                and self._since_partial_ms >= self.partial_interval_ms
            ):
                event = self._event("partial", frame_time)
                if event:
                    events.append(event)
                self._since_partial_ms = 0
        return events

    def flush(self) -> list[TranscriptEvent]:
        """Finalize speech at capture shutdown, including a short final PCM frame."""
        if self._pending and self._active:
            self._audio.extend(self._pending)
        self._pending.clear()
        if not self._active:
            return []
        timestamp_ms = self._last_capture_timestamp_ms
        if timestamp_ms is None:
            timestamp_ms = self._samples_seen * 1000 // SAMPLE_RATE
        return self._finish(timestamp_ms)
