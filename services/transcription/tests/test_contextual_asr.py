"""Audio context, timestamp commits, pause staging, and hallucination safeguards."""
import struct
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from echomind.asr import Transcriber
from echomind.cli import show_events
from echomind.asr import _decode_anomaly
from echomind.question import QuestionBuffer


def pcm(amplitude, ms):
    return struct.pack('<h', amplitude) * (16 * ms)


def recognized(*words):
    entries = [SimpleNamespace(start=start, end=end, word=text) for start, end, text in words]
    return [SimpleNamespace(text=''.join(w.word for w in entries), words=entries,
                            no_speech_prob=0.0, avg_logprob=-0.2)], None


def stream_with(*responses, **options):
    stream = Transcriber(context_window_ms=15000, partial_interval_ms=30000, **options)
    stream._model = Mock()
    stream._model.transcribe.side_effect = responses
    return stream


def test_partial_request_waits_and_continuation_is_decoded_with_original_audio():
    stream = stream_with(recognized((0.0, .3, '讲解一下React的')),
                         recognized((0.0, .3, '讲解一下React的'), (.9, 1.2, ' Hooks')))
    first = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text) for e in first] == [('partial', '讲解一下React的')]
    second = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text) for e in second] == [('final', '讲解一下React的 Hooks')]
    assert len(stream._model.transcribe.call_args.args[0]) == 1800 * 16


def test_complete_question_finishes_at_600ms_and_real_repetition_survives():
    stream = stream_with(recognized((0.0, .3, '解释一下Hooks')),
                         recognized((0.0, .3, '解释一下Hooks'), (.9, 1.2, '解释一下Hooks')))
    first = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    second = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.text, e.timestamp_ms) for e in first + second] == [('解释一下Hooks', 900), ('解释一下Hooks', 1800)]


def test_incomplete_candidate_times_out_without_redecoding_silence_or_auto_answer():
    stream = stream_with(recognized((0.0, .3, '讲解一下React的')))
    stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    events = stream.push_pcm(pcm(0, 1410))
    assert [(e.kind, e.text) for e in events] == [('final', '讲解一下React的')]
    stream._model.transcribe.assert_called_once()
    questions = QuestionBuffer()
    assert questions.push(events[0].text, events[0].timestamp_ms) is None
    assert questions.push('Hooks', 3000).text == '讲解一下React的Hooks'


def test_context_is_bounded_and_old_speech_is_not_emitted_during_silence():
    stream = stream_with(recognized((0.0, .3, '测试问题')))
    stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert stream.push_pcm(pcm(0, 30000)) == []
    assert len(stream._context) <= 15000 * 32
    stream._model.transcribe.assert_called_once()


def test_rolling_window_uses_absolute_time_after_trimming():
    stream = stream_with(recognized((0.0, .3, '前一个问题')),
                         recognized((14.1, 14.4, '后一个问题')))
    stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    stream.push_pcm(pcm(0, 15000))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [e.text for e in events] == ['后一个问题']
    assert len(stream._model.transcribe.call_args.args[0]) == 15000 * 16


@pytest.mark.parametrize('text', [
    '讲解一下Hooks, Hooks, Hooks, 灰度发布',
    'Hooks, React Fiber, Vue, JavaScript, Vue, JavaScript, Vue, JavaScript, Vue, JavaScript,',
    '。 。 。 。 。 。 。 。 。 这样自己稍微的Active Clips',
    '2-' * 35 + '2优优独播剧场——YoYo Television Series Exclusive',
    '12-34-' * 8,
])
def test_repetitive_output_is_preserved_but_never_sent_to_online_routing(text):
    stream = stream_with(recognized((0.0, .3, text)))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 2010))
    final = events[-1]
    assert len(events) == 1
    assert final.kind == 'rejected' and final.text == text and final.rejection_reason
    copilot = Mock()
    show_events(events, copilot)
    copilot.submit_transcript.assert_not_called()
    assert not any(stream._context)
    assert stream.push_pcm(pcm(0, 6000)) == []
    stream._model.transcribe.assert_called_once()


@pytest.mark.parametrize('text', [
    'React Hooks, React Fiber, Vue, JavaScript, TypeScript, ES6, Promise',
    'React和Vue有哪些区别？Vue和React该如何选择？',
    '为什么需要Hooks？如何使用Hooks？请举一个Hooks的例子。',
    '对对对，是这个意思。',
    '好好好，我们继续。',
    '1+1+1等于多少？',
    '2026-10-11，HTTP-2 和 React-Fiber',
])
def test_normal_technical_lists_and_explanations_are_not_rejected(text):
    assert not _decode_anomaly(text)


def test_abnormal_partial_is_not_shown_and_next_clean_utterance_recovers():
    stream = stream_with(recognized((0, .9, 'Vue, JavaScript, ' * 4)),
                         recognized((0, .3, '解释一下Hooks')))
    stream.partial_interval_ms = 700
    assert stream.push_pcm(pcm(800, 900)) == []
    assert stream.push_pcm(pcm(0, 600))[0].kind == 'rejected'
    stream.partial_interval_ms = 30000
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text) for e in events] == [('final', '解释一下Hooks')]
    assert stream._model.transcribe.call_count == 2


def test_rejected_preview_cannot_be_replaced_by_final_and_flush_clears_context():
    stream = stream_with(recognized((0, .9, 'React, Vue, ' * 12)),
                         recognized((0, .3, '欢迎订阅明镜')), retry_low_confidence=True)
    stream.partial_interval_ms = 700
    assert stream.push_pcm(pcm(800, 750)) == []
    assert stream.push_pcm(pcm(0, 300)) == []
    events = stream.flush()
    assert [event.kind for event in events] == ['rejected']
    stream._model.transcribe.assert_called_once()
    assert not stream._context and not stream._audio
    assert not stream._stable_partial_terms and not stream._previous_preview_terms
    assert stream._rejected_event is None
    assert stream.flush() == []


def test_new_speech_can_recover_from_a_bad_incomplete_preview():
    stream = stream_with(recognized((0, .7, 'Vue, JavaScript, ' * 4)),
                         recognized((0, 1.2, '为什么Redis比直接查询数据库快？')),
                         retry_low_confidence=True)
    stream.partial_interval_ms = 700
    assert stream.push_pcm(pcm(800, 750)) == []
    stream.partial_interval_ms = 30000
    events = stream.push_pcm(pcm(800, 450) + pcm(0, 600))
    assert [(e.kind, e.text) for e in events] == [('final', '为什么Redis比直接查询数据库快？')]
    assert stream._model.transcribe.call_count == 2


def test_rejected_long_turn_does_not_carry_failed_audio_into_next_turn():
    stream = stream_with(recognized((0, 1.2, '2-' * 30)),
                         recognized((0, .3, '解释一下Hooks')), max_utterance_ms=1200)
    events = stream.push_pcm(pcm(800, 1200))
    assert [event.kind for event in events] == ['rejected']
    assert not stream._active and not stream._audio and not stream._context
    assert not stream._carried_overlap
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text) for e in events] == [('final', '解释一下Hooks')]


def test_old_context_cannot_supply_words_for_a_new_noise_trigger():
    stream = stream_with(recognized((0, .3, '第一个问题')),
                         recognized((.4, .7, '旧音频补出来的内容')))
    stream.push_pcm(pcm(800, 300) + pcm(0, 6000))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert events == []  # Old words are filtered, not presented as current speech.


def test_previous_word_suffix_is_not_reattached_after_new_turn_starts():
    stream = stream_with(recognized((0, .3, '解释一下Hooks')),
                         recognized((.35, .6, 's'), (3.3, 3.6, '请解释Fiber')))
    stream.push_pcm(pcm(800, 300) + pcm(0, 3000))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [e.text for e in events] == ['请解释Fiber']


def test_design_prompt_waits_for_requirements_after_long_pause():
    stream = stream_with(recognized((0, .3, '设计一个灰度发布方案')),
                         recognized((0, .3, '设计一个灰度发布方案'), (1.6, 1.9, '支持按用户放量')))
    first = stream.push_pcm(pcm(800, 300) + pcm(0, 1290))
    assert [e.kind for e in first] == ['partial']
    second = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text) for e in second] == [('final', '设计一个灰度发布方案支持按用户放量')]


def test_design_without_continuation_finishes_with_one_decode():
    stream = stream_with(recognized((0, .3, '设计一个灰度发布方案')))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 2010))
    assert [e.kind for e in events] == ['partial', 'final']
    stream._model.transcribe.assert_called_once()


def test_word_alignment_drift_does_not_remove_first_word():
    stream = stream_with(recognized((1.0, 1.3, '讲解一下Hooks')))
    stream.push_pcm(pcm(0, 1800))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.kind, e.text) for e in events] == [('final', '讲解一下Hooks')]


def test_flush_commits_waiting_candidate_without_an_extra_decode():
    stream = stream_with(recognized((0.0, .3, '讲解一下')))
    stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert stream.flush()[0].text == '讲解一下'
    assert stream.flush() == []
    stream._model.transcribe.assert_called_once()


def test_alignment_cannot_advance_commit_past_audio_or_emit_future_words():
    stream = stream_with(recognized((0.0, 1.0, '测试问题'), (5.0, 6.0, '幻觉')))
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [e.text for e in events] == ['测试问题']
    assert stream._committed_until_ms == 900


def test_stable_preview_term_loss_redecodes_current_audio_not_text_splicing():
    stream = stream_with(recognized((0, .3, '请解释的握手')),
                         recognized((0, .3, '请解释TCP的握手')), retry_low_confidence=True)
    stream._stable_partial_terms = {'tcp'}
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert [(e.text, e.suspect, e.original_text) for e in events] == [
        ('请解释TCP的握手', False, '请解释的握手')]
    assert stream._model.transcribe.call_count == 2
    assert not stream._stable_partial_terms


def test_failed_term_recheck_preserves_original_for_manual_confirmation():
    stream = stream_with(recognized((0, .3, '请解释的握手')),
                         recognized((0, .3, '完全无关的字幕')), retry_low_confidence=True)
    stream._stable_partial_terms = {'tcp'}
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 2010))
    assert events[-1].text == '请解释的握手' and events[-1].suspect
    assert stream._model.transcribe.call_count == 2


def test_only_terms_in_two_consecutive_previews_become_stable():
    stream = stream_with(recognized((0, .3, 'React')),
                         recognized((0, .3, 'React')), recognized((0, .3, 'React Hooks')))
    stream._audio.extend(pcm(800, 300))
    stream._context.extend(stream._audio)
    stream._samples_seen = 4800
    stream._event('partial', 300)
    assert not stream._stable_partial_terms
    stream._event('partial', 300)  # Even a duplicate suppressed in the UI is evidence.
    assert stream._stable_partial_terms == {'react'}
    stream._event('partial', 300)
    assert stream._stable_partial_terms == {'react'}


@pytest.mark.parametrize('stable,text', [({'act', 'voo'}, 'React和Vue有什么区别'),
                                       ({'react', 'fiverr'}, '讲解React Fiber如何调度')])
def test_normal_spelling_revision_does_not_trigger_term_loss_review(stable, text):
    stream = stream_with(recognized((0, .3, text)), retry_low_confidence=True)
    stream._stable_partial_terms = stable
    events = stream.push_pcm(pcm(800, 300) + pcm(0, 600))
    assert events[0].text == text and not events[0].suspect
    stream._model.transcribe.assert_called_once()
