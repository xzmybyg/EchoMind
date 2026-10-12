"""Energy gate checks using synthetic PCM; no model or capture device needed."""

import struct
from unittest.mock import Mock

import numpy as np
import pytest

from echomind.vad import EnergyVAD, SpeechVAD


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


def test_speech_detector_buffers_30ms_frames_and_retains_recurrent_state():
    session = Mock()
    state = np.ones((1, 1, 128), dtype=np.float32)
    session.run.side_effect = [(np.array([0.8]), state, state), (np.array([0.4]), state, state), (np.array([0.2]), state, state)]
    gate = SpeechVAD(session)
    assert not gate.is_speech(frame(800))
    assert gate.is_speech(frame(800))
    assert gate.is_speech(frame(800))
    assert not gate.is_speech(frame(800))
    assert session.run.call_args_list[0].args[1]["input"].shape == (1, 576)
    assert np.all(session.run.call_args_list[0].args[1]["h"] == 0)
    assert np.all(session.run.call_args_list[1].args[1]["h"] == 1)
    assert np.all(SpeechVAD(session).h == 0)
    with pytest.raises(ValueError):
        gate.is_speech(b"x")


def test_real_silero_model_rejects_digital_silence_and_steady_tone():
    gate = SpeechVAD()
    assert all(not gate.is_speech(frame(0)) for _ in range(100))
    assert all(not gate.is_speech(frame(800)) for _ in range(100))
