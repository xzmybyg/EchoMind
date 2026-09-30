"""Streaming logic tests that never download or initialize a Whisper model."""

import struct

from echomind.asr import FRAME_BYTES, Transcriber, _join_without_overlap


def pcm(amplitude: int, milliseconds: int) -> bytes:
    return struct.pack("<h", amplitude) * (16 * milliseconds)


def test_partial_is_revised_and_finalized_after_silence() -> None:
    calls = 0

    def recognize(data: bytes) -> str:
        nonlocal calls
        calls += 1
        return "你好吗" if calls == 1 else "你好吗？"

    stream = Transcriber(recognizer=recognize)
    assert stream.push_pcm(pcm(800, 300)) == []
    partials = stream.push_pcm(pcm(800, 900))
    assert [event.kind for event in partials] == ["partial"]
    assert partials[0].text == "你好吗"
    finals = stream.push_pcm(pcm(0, 600))
    assert finals[-1].kind == "final"
    assert finals[-1].text == "你好吗？"
    assert finals[-1].timestamp_ms == 1800
    assert finals[-1].latency_ms >= 0


def test_silence_does_not_load_recognizer_or_emit_text() -> None:
    stream = Transcriber(recognizer=lambda _: (_ for _ in ()).throw(AssertionError()))
    assert stream.push_pcm(pcm(0, 1200)) == []
    assert stream.flush() == []


def test_short_sound_is_ignored() -> None:
    stream = Transcriber(recognizer=lambda _: "幻觉")
    assert stream.push_pcm(pcm(700, 120) + pcm(0, 600)) == []


def test_arbitrary_chunk_boundaries_and_flush() -> None:
    stream = Transcriber(recognizer=lambda _: "测试")
    audio = pcm(700, 420)
    assert stream.push_pcm(audio[:FRAME_BYTES + 2]) == []
    assert stream.push_pcm(audio[FRAME_BYTES + 2:], timestamp_ms=12_345) == []
    assert [(e.kind, e.text, e.timestamp_ms) for e in stream.flush()] == [
        ("final", "测试", 12_345)
    ]


def test_repeated_partials_are_suppressed() -> None:
    stream = Transcriber(recognizer=lambda _: "同一句")
    events = stream.push_pcm(pcm(700, 1800))
    assert [(e.kind, e.text) for e in events] == [("partial", "同一句")]


def test_long_speech_is_bounded_and_overlap_deduplicates() -> None:
    outputs = iter(["今天讨论方案", "论方案明天实施"])
    stream = Transcriber(
        recognizer=lambda _: next(outputs),
        partial_interval_ms=30_000,
        max_utterance_ms=1200,
    )
    first = stream.push_pcm(pcm(700, 1200))
    second = stream.push_pcm(pcm(700, 720))
    assert [(e.kind, e.text) for e in first + second] == [
        ("final", "今天讨论方案"),
        ("final", "明天实施"),
    ]
    assert len(stream._audio) <= 1200 * 32


def test_overlap_function_does_not_remove_single_matching_character() -> None:
    assert _join_without_overlap("今天讨论方案", "方案明天实施") == "明天实施"
    assert _join_without_overlap("今天", "天晴了") == "天晴了"
