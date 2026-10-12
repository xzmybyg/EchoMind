"""Energy and streaming speech detectors for 16 kHz mono int16 PCM."""

from __future__ import annotations

from array import array
from math import sqrt
import sys


class EnergyVAD:
    def __init__(self, min_rms: float = 60.0, noise_multiplier: float = 2.5) -> None:
        self.min_rms = min_rms
        self.noise_multiplier = noise_multiplier
        self.noise_rms = 30.0

    def is_speech(self, pcm: bytes) -> bool:
        if len(pcm) % 2:
            raise ValueError("PCM frame must contain complete int16 samples")
        if not pcm:
            return False
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        rms = sqrt(sum(sample * sample for sample in samples) / len(samples))
        speech = rms >= max(self.min_rms, self.noise_rms * self.noise_multiplier)
        if not speech:
            self.noise_rms = 0.98 * self.noise_rms + 0.02 * rms
        return speech


class SpeechVAD:
    """Streaming Silero v6 detector with session-local recurrent state (16 kHz)."""

    def __init__(self, session=None):
        import numpy as np
        if session is None:
            from faster_whisper.vad import get_vad_model
            session = get_vad_model().session
        self.session = session
        self.pending = bytearray()
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.speech = False

    def is_speech(self, pcm: bytes) -> bool:
        import numpy as np
        if len(pcm) % 2:
            raise ValueError("PCM frame must contain complete int16 samples")
        self.pending.extend(pcm)
        while len(self.pending) >= 1024:
            samples = np.frombuffer(bytes(self.pending[:1024]), dtype="<i2").astype(np.float32) / 32768.0
            del self.pending[:1024]
            audio = np.concatenate((self.context, samples.reshape(1, -1)), axis=1)
            probability, self.h, self.c = self.session.run(None, {"input": audio, "h": self.h, "c": self.c})
            self.context = samples[-64:].reshape(1, -1)
            self.speech = float(np.asarray(probability).reshape(-1)[0]) >= (0.35 if self.speech else 0.5)
        return self.speech
