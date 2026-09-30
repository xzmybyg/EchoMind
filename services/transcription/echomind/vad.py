"""Small, dependency-free voice activity gate for 16 kHz PCM frames.

This is a deliberately modest first-stage gate, not a speech classifier. The
threshold is exposed so a quiet Tencent Meeting output can be calibrated.
"""

from __future__ import annotations

from array import array
from math import sqrt
import sys


class EnergyVAD:
    def __init__(self, min_rms: float = 90.0, noise_multiplier: float = 2.5) -> None:
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
