"""Default microphone pipeline tests; fake devices never record real people."""

from types import SimpleNamespace
import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from echomind.asr import TranscriptEvent
from echomind.microphone import MicrophoneError, input_quality, run_microphone


def run(stopping, callback=lambda _: None, copilot=None):
    run_microphone("model", "cpu", "int8", copilot, stop_event=stopping,
                   on_transcript=callback, on_status=lambda _: None)


@pytest.mark.parametrize("amplitude,expected", [(0, "未检测到"), (40, "输入较低"),
                                               (1000, ""), (32767, "削波"), (-32768, "削波")])
def test_input_quality_distinguishes_silence_weak_input_and_clipping(amplitude, expected):
    level, message = input_quality(np.full(1600, amplitude, dtype="<i2").tobytes())
    assert expected in message
    assert level == pytest.approx(abs(amplitude) / 32768)


@pytest.fixture
def model():
    with patch("echomind.asr.Transcriber") as factory:
        factory.return_value.push_pcm.return_value = []
        yield factory.return_value


@pytest.mark.parametrize('kind', ['final', 'rejected'])
def test_default_microphone_resamples_and_routes_final_to_answers(model, kind):
    stopping = threading.Event()
    final = TranscriptEvent(kind, "Redis为什么快", 1234, 1, suspect=True)
    model.push_pcm.return_value = [final]
    copilot = Mock()
    closed = []
    received = []

    def open_stream(**kwargs):
        assert kwargs["samplerate"] == 48000
        assert kwargs["channels"] == 1
        assert kwargs["dtype"] == "int16"
        def start():
            kwargs["callback"](np.full(4800, 1000, dtype="<i2").tobytes(), 4800, None, SimpleNamespace(input_overflow=False))
        return SimpleNamespace(start=start, close=lambda: closed.append(True), active=True)

    def transcript(event):
        received.append(event)
        stopping.set()

    with patch("sounddevice.query_devices", return_value={"default_samplerate": 48000, "max_input_channels": 2}), patch("sounddevice.RawInputStream", side_effect=open_stream):
        run(stopping, transcript, copilot)
    assert received == [final]
    pcm = model.push_pcm.call_args.args[0]
    assert 3000 <= len(pcm) <= 3200 and len(pcm) % 2 == 0
    if kind == 'rejected':
        copilot.submit_transcript.assert_not_called()
    else:
        copilot.submit_transcript.assert_called_once_with(final.text, final.timestamp_ms, final=True, suspect=True)
    assert closed == [True]


def test_stop_after_warmup_never_opens_microphone(model):
    stopping = threading.Event()
    model.warm_up.side_effect = stopping.set
    with patch("sounddevice.RawInputStream") as stream:
        run(stopping)
        stream.assert_not_called()


def test_missing_device_is_safe_and_actionable(model):
    with patch("sounddevice.query_devices", side_effect=RuntimeError("private-driver-data")):
        with pytest.raises(MicrophoneError, match="默认输入设备") as caught:
            run(threading.Event())
    assert "private-driver-data" not in str(caught.value)


def test_failed_stream_start_closes_device(model):
    stream = Mock()
    stream.start.side_effect = RuntimeError("private-driver-data")
    with patch("sounddevice.query_devices", return_value={"default_samplerate": 48000, "max_input_channels": 1}), patch("sounddevice.RawInputStream", return_value=stream):
        with pytest.raises(MicrophoneError, match="麦克风权限"):
            run(threading.Event())
    stream.close.assert_called_once()


@pytest.mark.parametrize("overflow,packets", [(True, 1), (False, 301)])
def test_audio_overflow_is_reported_and_stream_closed(model, overflow, packets):
    closed = []
    def open_stream(**kwargs):
        def start():
            for _ in range(packets):
                kwargs["callback"](b"\0" * 9600, 4800, None, SimpleNamespace(input_overflow=overflow))
        return SimpleNamespace(start=start, close=lambda: closed.append(True), active=True)
    with patch("sounddevice.query_devices", return_value={"default_samplerate": 48000, "max_input_channels": 1}), patch("sounddevice.RawInputStream", side_effect=open_stream):
        with pytest.raises(MicrophoneError, match="丢失|超过 30 秒"):
            run(threading.Event())
    assert closed == [True]


def test_disconnect_is_detected_and_closed(model):
    stream = Mock(active=False)
    with patch("sounddevice.query_devices", return_value={"default_samplerate": 48000, "max_input_channels": 1}), patch("sounddevice.RawInputStream", return_value=stream):
        with pytest.raises(MicrophoneError, match="断开"):
            run(threading.Event())
    stream.close.assert_called_once()


def test_backlog_beyond_ten_seconds_is_processed_without_dropping_audio(model):
    stopping = threading.Event()
    closed = []

    def process(*args, **kwargs):
        if model.push_pcm.call_count == 150:
            stopping.set()
        return []

    model.push_pcm.side_effect = process

    def open_stream(**kwargs):
        def start():
            for _ in range(150):  # 15 s queued before recognition resumes.
                kwargs["callback"](b"\0" * 9600, 4800, None, SimpleNamespace(input_overflow=False))
        return SimpleNamespace(start=start, close=lambda: closed.append(True), active=True)

    with patch("sounddevice.query_devices", return_value={"default_samplerate": 48000, "max_input_channels": 1}), patch("sounddevice.RawInputStream", side_effect=open_stream):
        run(stopping)
    assert model.push_pcm.call_count == 150
    assert closed == [True]
