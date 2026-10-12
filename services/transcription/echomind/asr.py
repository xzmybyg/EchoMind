"""Synchronous, incremental Chinese transcription from 16 kHz mono PCM.

``push_pcm`` accepts arbitrary sized little-endian int16 chunks. It can block
while Whisper runs, so callers capturing audio should feed it from a worker.
``timestamp_ms`` is the capture time of the chunk's end; if omitted, events use
the audio stream's elapsed time. ``latency_ms`` measures ASR processing time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Literal
from difflib import SequenceMatcher
import re

from .vad import EnergyVAD, SpeechVAD
from .text import normalize_transcript
from .question import is_unfinished as _unfinished


SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2
PREVIOUS_DEFAULT_HOTWORDS = "React Fiber, Promise, Redis, TCP, HTTP, JavaScript, TypeScript, 灰度发布"
LEGACY_DEFAULT_HOTWORDS = "React Hooks, Hooks, " + PREVIOUS_DEFAULT_HOTWORDS
DEFAULT_HOTWORDS = "React Hooks, React Fiber, Vue, JavaScript, TypeScript, ES6, Promise"


def _repetitive(text: str) -> bool:
    # Preserve the words for review; repeated technical words are not auto-corrected.
    tokens = re.findall(r"[a-z_][a-z0-9_'-]*|[0-9]+|[\u4e00-\u9fff]", text.lower())
    for size in range(1, min(8, len(tokens) // 3) + 1):
        for start in range(len(tokens) - size * 3 + 1):
            unit = tokens[start:start + size]
            # Short acknowledgements and arithmetic (对对对, 1+1+1) are valid.
            repeats = 6 if size == 1 and len(unit[0]) == 1 else 3
            if tokens[start:start + size * repeats] == unit * repeats:
                return True
    return False


def _decode_anomaly(text: str) -> str:
    if _repetitive(text):
        return "识别结果出现循环重复"
    if re.search(r"(?:[。.!！?？,，;；、…·]\s*){8,}", text):
        return "识别结果出现异常连续标点"
    return ""


def _needs_continuation(text: str) -> bool:
    # Task prompts often put requirements after a pause following “方案”.
    return _unfinished(text) or bool(re.search(
        r"^(?:请|帮我|请帮我)?(?:设计|实现|制定).*(?:方案|系统)[。！？?！\s]*$", text))


def _technical_terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9]*", text.lower()) if len(word) >= 3}


def _lost_preview_terms(stable: set[str], text: str) -> bool:
    current = _technical_terms(text)
    # Spelling revisions (Voo -> Vue, Fiverr -> Fiber) are normal. Only
    # disappearing slots or clipped suffixes warrant another audio decode.
    return len(current) < len(stable) or any(
        old != new and old.endswith(new) for old in stable for new in current)


@dataclass(frozen=True)
class TranscriptEvent:
    kind: Literal["partial", "final", "rejected"]
    text: str
    timestamp_ms: int
    latency_ms: float
    suspect: bool = False
    original_text: str = ""
    rejection_reason: str = ""


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
        vad: EnergyVAD | SpeechVAD | None = None,
        partial_interval_ms: int = 700,
        min_speech_ms: int = 180,
        end_silence_ms: int = 600,
        max_utterance_ms: int = 12_000,
        hotwords: str | None = None,
        retry_low_confidence: bool = False,
        context_window_ms: int = 0,
        finalize_delay_ms: int = 2000,
        wait_for_continuation: bool = False,
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
        # Optional short terms, never a natural-language instruction or sentence.
        self.hotwords = hotwords
        self.retry_low_confidence = retry_low_confidence
        self._last_suspect = False
        self._last_original = ""
        self._last_rejection = ""
        self._utterance_start_ms = 0.0
        self.context_window_ms = context_window_ms
        self.finalize_delay_ms = max(end_silence_ms, finalize_delay_ms)
        self.wait_for_continuation = wait_for_continuation
        self._context = bytearray()
        self._committed_until_ms = 0.0
        self._last_word_end_ms = 0.0
        self._candidate: TranscriptEvent | None = None
        self._candidate_checked = False
        self._rejected_event: TranscriptEvent | None = None
        self._rejected_voice_ms = 0

        self._pending = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=16)  # 480 ms, including delayed VAD onset.
        self._audio = bytearray()
        self._active = False
        self._voice_ms = 0
        self._silence_ms = 0
        self._since_partial_ms = 0
        self._last_partial = ""
        self._stable_partial_terms: set[str] = set()
        self._previous_preview_terms: set[str] = set()
        self._previous_final = ""
        self._carried_overlap = False
        self._samples_seen = 0
        self._last_capture_timestamp_ms: int | None = None

    def warm_up(self) -> None:
        """Load the model and complete its first inference before capture starts."""
        if self._recognizer is None:
            self._transcribe(bytes(SAMPLE_RATE * 2), warmup=True)

    def _transcribe(
        self, pcm: bytes, *, final: bool = False, warmup: bool = False,
        skip_before_ms: float | None = None, retry_allowed: bool = True,
    ) -> tuple[str, float]:
        started = perf_counter()
        self._last_suspect = False
        self._last_original = ""
        self._last_rejection = ""
        self._last_word_end_ms = 0.0
        # Warmup must still initialize/infer; real digital silence must not decode.
        if not warmup and not any(pcm):
            return "", (perf_counter() - started) * 1000
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
            # Only boost quiet, gated utterances; cap gain to avoid magnifying noise.
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            if 0.0 < peak < 0.1:
                samples *= min(8.0, 0.1 / peak)

            def decode(beam, hotwords):
                timing = {"word_timestamps": True} if skip_before_ms is not None else {}
                segments, _ = self._model.transcribe(
                    samples, language="zh", beam_size=beam, temperature=0.0,
                    condition_on_previous_text=False, vad_filter=False,
                    hotwords=hotwords,
                    **timing,
                )
                texts, scores, suspect = [], [], False
                rejection, word_end = "", 0.0
                for segment in segments:
                    no_speech = getattr(segment, "no_speech_prob", 0.0)
                    logprob = getattr(segment, "avg_logprob", 0.0)
                    if no_speech > 0.6 and logprob <= -1.0:
                        continue
                    text = segment.text
                    rejection = rejection or _decode_anomaly(text)
                    if skip_before_ms is not None:
                        words = [word for word in (getattr(segment, "words", None) or [])
                                 if 0 <= word.start <= word.end and word.start * 1000 <= len(pcm) / 32
                                 and word.end * 1000 > skip_before_ms]
                        if not words:
                            continue
                        text = "".join(word.word for word in words)
                        word_end = max(word_end, min(words[-1].end * 1000, len(pcm) / 32))
                    suspect |= no_speech > 0.5 or logprob < -0.8
                    texts.append(text)
                    scores.append(logprob)
                    if rejection:
                        break  # Do not keep decoding more segments of a detected loop.
                text = normalize_transcript("".join(texts))
                return text, min(scores, default=-10.0), suspect, rejection or _decode_anomaly(text), word_end

            result, score, self._last_suspect, self._last_rejection, self._last_word_end_ms = decode(
                5 if final else 1, self.hotwords if final else None)
            if (final and retry_allowed and self.retry_low_confidence
                    and self._last_suspect and not self._last_rejection and result):
                revised, revised_score, revised_suspect, revised_rejection, revised_end = decode(8, None)
                # Wider search is evidence, not permission to replace a different sentence.
                similar = SequenceMatcher(None, result, revised).ratio() >= 0.55
                if (revised and not revised_suspect and not revised_rejection
                        and revised_score >= score + 0.15 and similar):
                    if revised != result:
                        self._last_original = result
                    self._last_suspect = False
                    result, self._last_rejection, self._last_word_end_ms = revised, "", revised_end
        result = normalize_transcript(result)
        self._last_rejection = self._last_rejection or _decode_anomaly(result)
        self._last_suspect |= bool(self._last_rejection)
        return result, (perf_counter() - started) * 1000

    def _event(self, kind: Literal["partial", "final"], timestamp_ms: int) -> TranscriptEvent | None:
        # Silence/retries are not new evidence. Only new speech can reopen a preview.
        if self._rejected_event is not None:
            if self._voice_ms - self._rejected_voice_ms < self.min_speech_ms:
                return replace(self._rejected_event, timestamp_ms=timestamp_ms) if kind == "final" else None
            self._rejected_event = None
        if self.context_window_ms:
            context_start_ms = (self._samples_seen - len(self._context) // 2) * 1000 / SAMPLE_RATE
            boundary = max(self._committed_until_ms, self._utterance_start_ms - 800)
            text, latency_ms = self._transcribe(bytes(self._context), final=kind == "final",
                                               skip_before_ms=max(0, boundary - context_start_ms))
            self._last_word_end_ms += context_start_ms
            # Alignment can shift between decodes. Allow 800 ms of drift, but
            # never present words entirely before this speech turn as new speech.
            if text and self._recognizer is None and self._last_word_end_ms < self._utterance_start_ms - 800:
                self._last_rejection = "识别文字未对应到本次发言"
        else:
            text, latency_ms = self._transcribe(bytes(self._audio), final=kind == "final")
        if (kind == "final" and self.retry_low_confidence and self.context_window_ms
                and self._model is not None and not self._last_suspect and not self._last_rejection
                and not self._last_original and _lost_preview_terms(self._stable_partial_terms, text)):
            # A term heard in two consecutive previews disappeared from the final.
            # Re-decode the actual current turn, never splice preview text into it.
            previous, previous_end = text, self._last_word_end_ms
            revised, elapsed = self._transcribe(bytes(self._audio), final=True,
                                               skip_before_ms=0, retry_allowed=False)
            latency_ms += elapsed
            if (revised and not self._last_suspect and not self._last_rejection
                    and not _lost_preview_terms(self._stable_partial_terms, revised)
                    and (self._stable_partial_terms - _technical_terms(previous)) & _technical_terms(revised)
                    and SequenceMatcher(None, previous, revised).ratio() >= .55):
                text = revised
                self._last_original = previous if previous != revised else ""
                self._last_word_end_ms += (self._samples_seen - len(self._audio) // 2) * 1000 / SAMPLE_RATE
            else:
                text = previous
                self._last_suspect = True
                self._last_original = self._last_rejection = ""
                self._last_word_end_ms = previous_end
        if self._carried_overlap and not self.context_window_ms:
            text = _join_without_overlap(self._previous_final, text)
        if kind == "partial":
            terms = _technical_terms(text) if text and not self._last_rejection and not self._last_suspect else set()
            shared = self._previous_preview_terms & terms
            if shared:
                self._stable_partial_terms = shared
            self._previous_preview_terms = terms
        if self._last_rejection:
            self._rejected_event = TranscriptEvent("rejected", text, timestamp_ms, latency_ms, True,
                                                   rejection_reason=self._last_rejection)
            self._rejected_voice_ms = self._voice_ms
            return self._rejected_event if kind == "final" else None
        if not text or (kind == "partial" and text == self._last_partial):
            return None
        if kind == "partial":
            self._last_partial = text
        elif not self.context_window_ms:
            self._previous_final = (self._previous_final + text)[-200:]
        return TranscriptEvent(kind, text, timestamp_ms, latency_ms, self._last_suspect, self._last_original)

    def _finish(self, timestamp_ms: int, *, carry: bool = False) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        if self._voice_ms >= self.min_speech_ms:
            event = (replace(self._candidate, timestamp_ms=timestamp_ms) if self._candidate is not None
                     else None if self._candidate_checked else self._event("final", timestamp_ms))
            if event:
                events.append(event)
        if self.context_window_ms:
            self._committed_until_ms = max(self._committed_until_ms, self._last_word_end_ms)
        if events and events[-1].kind == "rejected":
            # Do not carry the same failed decode's audio into another attempt.
            self._context.clear()
            self._previous_final = ""
            self._committed_until_ms = self._samples_seen * 1000 / SAMPLE_RATE
            carry = False
        self._rejected_event = None
        self._candidate = None
        self._candidate_checked = False
        tail = bytes(self._audio[-SAMPLE_RATE:]) if carry else b""  # 500 ms
        self._audio = bytearray(tail)
        self._active = carry
        self._voice_ms = 0
        self._silence_ms = 0
        self._since_partial_ms = 0
        self._last_partial = ""
        self._stable_partial_terms.clear()
        self._previous_preview_terms.clear()
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
            if self.context_window_ms:
                self._context.extend(frame)
                limit = self.context_window_ms * SAMPLE_RATE * 2 // 1000
                if len(self._context) > limit:
                    del self._context[:-limit]
            frame_time = self._samples_seen * 1000 // SAMPLE_RATE
            if timestamp_ms is not None:
                frame_time = timestamp_ms - len(self._pending) * 1000 // (SAMPLE_RATE * 2)

            speech = self.vad.is_speech(frame)
            if not self._active:
                self._preroll.append(frame)
                if speech:
                    self._active = True
                    self._utterance_start_ms = self._samples_seen * 1000 / SAMPLE_RATE - FRAME_MS
                    self._audio.extend(b"".join(self._preroll))
                    self._preroll.clear()
                    self._voice_ms = FRAME_MS
                continue

            self._audio.extend(frame)
            self._since_partial_ms += FRAME_MS
            if speech:
                self._voice_ms += FRAME_MS
                self._silence_ms = 0
                self._candidate = None
                self._candidate_checked = False
            else:
                self._silence_ms += FRAME_MS

            if self._silence_ms >= self.end_silence_ms:
                if (self.context_window_ms or self.wait_for_continuation) and self._voice_ms >= self.min_speech_ms:
                    if not self._candidate_checked:
                        self._candidate = self._event("final", frame_time)
                        self._candidate_checked = True
                        if self._candidate and self._candidate.kind != "rejected" and (_needs_continuation(self._candidate.text) or self._candidate.suspect):
                            events.append(replace(self._candidate, kind="partial"))
                    waiting = self._candidate and self._candidate.kind != "rejected" and (_needs_continuation(self._candidate.text) or self._candidate.suspect)
                    if waiting and self._silence_ms < self.finalize_delay_ms:
                        continue
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
            if self.context_window_ms:
                self._context.extend(self._pending)
                self._samples_seen += len(self._pending) // 2
                limit = self.context_window_ms * SAMPLE_RATE * 2 // 1000
                del self._context[:-limit]
                self._candidate = None
                self._candidate_checked = False
        self._pending.clear()
        if not self._active:
            return []
        timestamp_ms = self._last_capture_timestamp_ms
        if timestamp_ms is None:
            timestamp_ms = self._samples_seen * 1000 // SAMPLE_RATE
        return self._finish(timestamp_ms)
