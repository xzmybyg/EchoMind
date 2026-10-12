"""Streaming logic tests that never download or initialize a Whisper model."""

import struct
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from echomind.asr import FRAME_BYTES, TranscriptEvent, Transcriber, _join_without_overlap
from echomind.text import normalize_transcript


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


def test_quiet_short_utterance_is_finalized() -> None:
    recognize = Mock(return_value="一加二等于三")
    stream = Transcriber(recognizer=recognize, partial_interval_ms=30_000)
    events = stream.push_pcm(pcm(80, 180) + pcm(0, 600))
    assert [(event.kind, event.text) for event in events] == [
        ("final", "一加二等于三")
    ]
    recognize.assert_called_once()


def test_low_level_background_does_not_call_recognizer() -> None:
    recognize = Mock(side_effect=AssertionError("Background must stay gated"))
    stream = Transcriber(recognizer=recognize)
    assert stream.push_pcm(pcm(40, 1800)) == []
    assert stream.flush() == []
    recognize.assert_not_called()


def test_whisper_final_uses_wider_beam_than_partial_and_warmup() -> None:
    stream = Transcriber()
    model = Mock()
    model.transcribe.side_effect = lambda *_args, **_kwargs: (
        [SimpleNamespace(text="测试")], None
    )
    stream._model = model
    stream.warm_up()
    stream.push_pcm(pcm(800, 900) + pcm(0, 600))
    beams = [call.kwargs["beam_size"] for call in model.transcribe.call_args_list]
    assert len(beams) >= 3  # Warmup, one or more partials, and final.
    assert beams[:-1] == [1] * (len(beams) - 1)
    assert beams[-1] == 5
    for call in model.transcribe.call_args_list:
        assert call.kwargs["language"] == "zh"
        assert "initial_prompt" not in call.kwargs
        assert call.kwargs["temperature"] == 0.0
        assert call.kwargs["hotwords"] is None
        assert call.kwargs["condition_on_previous_text"] is False
        assert call.kwargs["vad_filter"] is False


def test_normalize_transcript_uses_simplified_chinese_without_rewriting_english() -> None:
    assert normalize_transcript("  解釋一下 React-Fiber 原理與資料庫設計  ") == (
        "解释一下 React-Fiber 原理与资料库设计"
    )
    assert normalize_transcript("React Library 與 React Vibre") == (
        "React Library 与 React Vibre"
    )
    assert normalize_transcript(" ") == ""


def test_injected_recognizer_normalizes_partial_and_final() -> None:
    stream = Transcriber(recognizer=lambda _: "解釋一下 React-Fiber 原理")
    events = stream.push_pcm(pcm(800, 900) + pcm(0, 600))
    assert [event.kind for event in events] == ["partial", "final"]
    assert all(event.text == "解释一下 React-Fiber 原理" for event in events)


def test_whisper_result_normalizes_before_events() -> None:
    stream = Transcriber()
    stream._model = Mock()
    stream._model.transcribe.return_value = (
        [SimpleNamespace(text="請講解一下"), SimpleNamespace(text=" React Fiber 原理")],
        None,
    )
    assert stream._transcribe(pcm(800, 300), final=True)[0] == (
        "请讲解一下 React Fiber 原理"
    )


def test_normalized_duplicate_partials_are_suppressed() -> None:
    outputs = iter(["解釋一下原理", "解释一下原理"])
    stream = Transcriber(recognizer=lambda _: next(outputs))
    events = stream.push_pcm(pcm(700, 1800))
    assert [(event.kind, event.text) for event in events] == [
        ("partial", "解释一下原理")
    ]


def test_whisper_quiet_audio_gain_is_capped_and_silence_stays_zero() -> None:
    stream = Transcriber()
    stream._model = Mock()
    stream._model.transcribe.return_value = ([], None)
    for amplitude, expected_peak in [(80, 640 / 32768), (1000, 0.1), (8000, 8000 / 32768)]:
        stream._transcribe(pcm(amplitude, 180), final=True)
        samples = stream._model.transcribe.call_args.args[0]
        assert samples.dtype == np.float32
        assert np.isclose(np.max(np.abs(samples)), expected_peak)


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


def test_digital_silence_does_not_decode_prompt_hallucination() -> None:
    stream = Transcriber()
    stream._model = Mock()
    stream._model.transcribe.return_value = (
        [SimpleNamespace(text="简体中文技术讨论。")], None
    )
    assert stream._transcribe(pcm(0, 1200), final=True)[0] == ""
    stream._model.transcribe.assert_not_called()
    stream.warm_up()
    stream._model.transcribe.assert_called_once()
    assert not np.any(stream._model.transcribe.call_args.args[0])
    assert "initial_prompt" not in stream._model.transcribe.call_args.kwargs


def test_optional_hotwords_are_used_only_for_final() -> None:
    stream = Transcriber(hotwords="React Fiber Redis")
    stream._model = Mock()
    stream._model.transcribe.return_value = ([], None)
    stream.warm_up()
    stream._transcribe(pcm(800, 300))
    stream._transcribe(pcm(800, 300), final=True)
    assert [call.kwargs["hotwords"] for call in stream._model.transcribe.call_args_list] == [
        None, None, "React Fiber Redis"
    ]


def test_joint_low_confidence_and_no_speech_segment_is_removed() -> None:
    stream = Transcriber(partial_interval_ms=30_000)
    stream._model = Mock()
    stream._model.transcribe.return_value = (
        [
            SimpleNamespace(text="简体中文技术讨论。", no_speech_prob=0.95, avg_logprob=-1.4),
            SimpleNamespace(text="一加一等于几？", no_speech_prob=0.1, avg_logprob=-0.3),
        ], None,
    )
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text, e.suspect) for e in events] == [
        ("final", "一加一等于几？", False)
    ]


def test_low_likelihood_speech_is_preserved_and_marked_suspect() -> None:
    stream = Transcriber(partial_interval_ms=30_000)
    stream._model = Mock()
    stream._model.transcribe.return_value = (
        [SimpleNamespace(text="1加1等于进。", no_speech_prob=0.1, avg_logprob=-1.2)], None
    )
    events = stream.push_pcm(pcm(80, 180) + pcm(0, 600))
    assert [(e.text, e.suspect) for e in events] == [("1加1等于进。", True)]


def test_high_silence_probability_does_not_discard_likely_short_speech() -> None:
    stream = Transcriber(partial_interval_ms=30_000)
    stream._model = Mock()
    stream._model.transcribe.return_value = (
        [SimpleNamespace(text="几？", no_speech_prob=0.8, avg_logprob=-0.3)], None
    )
    events = stream.push_pcm(pcm(80, 180) + pcm(0, 600))
    assert [(e.text, e.suspect) for e in events] == [("几？", True)]


def test_suspect_is_reset_between_utterances_and_old_event_constructor_works() -> None:
    stream = Transcriber(partial_interval_ms=30_000)
    stream._model = Mock()
    stream._model.transcribe.side_effect = [
        ([SimpleNamespace(text="含混问题", no_speech_prob=0.1, avg_logprob=-1.2)], None),
        ([SimpleNamespace(text="清楚问题", no_speech_prob=0.1, avg_logprob=-0.3)], None),
    ]
    first = stream.push_pcm(pcm(800, 180) + pcm(0, 600))
    second = stream.push_pcm(pcm(800, 180) + pcm(0, 600))
    assert first[0].suspect is True
    assert second[0].suspect is False
    assert TranscriptEvent("final", "旧构造", 0, 0).suspect is False


def test_short_pause_keeps_both_parts_in_one_utterance():
    recognize = Mock(return_value="设计一个灰度发布方案，支持按用户逐步放量")
    stream = Transcriber(recognizer=recognize, end_silence_ms=900, partial_interval_ms=30_000)
    assert stream.push_pcm(pcm(800, 300) + pcm(0, 600)) == []
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 900))
    assert len(events) == 1
    assert events[0].kind == "final"
    recognize.assert_called_once()
    assert len(recognize.call_args.args[0]) >= len(pcm(800, 1500))


def test_low_confidence_audio_retry_preserves_original_and_resets_next_time():
    stream = Transcriber(retry_low_confidence=True)
    stream._model = Mock()
    stream._model.transcribe.side_effect = [
        ([SimpleNamespace(text="解释React Fibr原理", avg_logprob=-0.9)], None),
        ([SimpleNamespace(text="解释React Fiber原理", avg_logprob=-0.4)], None),
        ([SimpleNamespace(text="下一题", avg_logprob=-0.4)], None),
    ]
    text, _ = stream._transcribe(pcm(800, 500), final=True)
    assert text == "解释React Fiber原理"
    assert stream._last_original == "解释React Fibr原理"
    assert not stream._last_suspect
    assert [call.kwargs["beam_size"] for call in stream._model.transcribe.call_args_list] == [5, 8]
    stream._transcribe(pcm(800, 500), final=True)
    assert stream._last_original == ""


def test_audio_retry_does_not_replace_with_unrelated_or_still_uncertain_text():
    for revised, score in (("一个完全无关的句子", -0.3), ("解释React Fiber原理", -0.85)):
        stream = Transcriber(retry_low_confidence=True)
        stream._model = Mock()
        stream._model.transcribe.side_effect = [
            ([SimpleNamespace(text="解释React Fibr原理", avg_logprob=-0.9)], None),
            ([SimpleNamespace(text=revised, avg_logprob=score)], None),
        ]
        text, _ = stream._transcribe(pcm(800, 500), final=True)
        assert text == "解释React Fibr原理"
        assert stream._last_suspect
        assert stream._last_original == ""


def test_confident_final_and_suspect_partial_do_not_retry():
    stream = Transcriber(retry_low_confidence=True)
    stream._model = Mock()
    stream._model.transcribe.return_value = ([SimpleNamespace(text="问题", avg_logprob=-0.9)], None)
    stream._transcribe(pcm(800, 500))
    stream._model.transcribe.assert_called_once()
    stream._model.reset_mock()
    stream._model.transcribe.return_value = ([SimpleNamespace(text="问题", avg_logprob=-0.3)], None)
    stream._transcribe(pcm(800, 500), final=True)
    stream._model.transcribe.assert_called_once()


def test_repetition_is_rejected_without_retrying_into_unrelated_captions():
    stream = Transcriber(retry_low_confidence=True, hotwords="React Fiber")
    stream._model = Mock()
    stream._model.transcribe.side_effect = [
        ([SimpleNamespace(text="React Fiber, " * 12, avg_logprob=-.6)], None),
        ([SimpleNamespace(text="欢迎订阅明镜", avg_logprob=-.2)], None),
    ]
    text, _ = stream._transcribe(pcm(800, 500), final=True)
    assert text == ("React Fiber, " * 12).strip()
    assert stream._last_suspect and stream._last_rejection
    assert not stream._last_original
    stream._model.transcribe.assert_called_once()


def test_repetition_across_segments_is_rejected_before_retry():
    stream = Transcriber(retry_low_confidence=True)
    stream._model = Mock()
    stream._model.transcribe.return_value = (
        [SimpleNamespace(text="React, Vue, ", avg_logprob=-.9) for _ in range(4)], None)
    stream._audio.extend(pcm(800, 500))
    event = stream._event("final", 500)
    assert event.kind == "rejected"
    stream._model.transcribe.assert_called_once()


def test_failed_retry_does_not_replace_first_pass_alignment_or_error():
    stream = Transcriber(retry_low_confidence=True)
    stream._model = Mock()
    stream._model.transcribe.side_effect = [
        ([SimpleNamespace(text="低可信问题", avg_logprob=-.9, words=[SimpleNamespace(start=0, end=.3, word="低可信问题")])], None),
        ([SimpleNamespace(text="无关无关无关", avg_logprob=-.1, words=[SimpleNamespace(start=.3, end=.5, word="无关无关无关")])], None),
    ]
    text, _ = stream._transcribe(pcm(800, 500), final=True, skip_before_ms=0)
    assert text == "低可信问题" and stream._last_suspect
    assert stream._last_word_end_ms == 300
    assert not stream._last_rejection
