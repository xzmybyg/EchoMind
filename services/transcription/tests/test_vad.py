"""Energy gate checks using synthetic PCM; no model or capture device needed."""

import struct

from echomind.vad import EnergyVAD


def frame(amplitude: int) -> bytes:
    return struct.pack("<h", amplitude) * 480


def test_quiet_frame_above_adaptive_floor_is_speech() -> None:
    assert EnergyVAD().is_speech(frame(80))


def test_silence_and_low_background_stay_gated() -> None:
    gate = EnergyVAD()
    assert not gate.is_speech(frame(0))
    assert all(not gate.is_speech(frame(40)) for _ in range(100))


def test_background_adaptation_still_raises_threshold() -> None:
    gate = EnergyVAD()
    assert all(not gate.is_speech(frame(70)) for _ in range(100))
    assert gate.noise_rms > 60
    assert not gate.is_speech(frame(100))
    assert gate.is_speech(frame(200))
