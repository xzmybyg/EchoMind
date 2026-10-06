"""Background generation and cancellation without models or network calls."""

import asyncio
import threading
import pytest
from types import SimpleNamespace

from echomind.copilot import Copilot
from echomind.llm import AnswerProviderError
from echomind.intent import QuestionGateError


class EveryFinal:
    def push(self, text, timestamp_ms, final=True):
        return SimpleNamespace(text=text) if final else None


def test_manual_question_bypasses_buffer_and_gate_and_allows_repeated_topics():
    requests = []
    events = []

    class RejectBuffer:
        def push(self, *args, **kwargs):
            raise AssertionError("manual question must bypass rules")

    class RejectGate:
        async def evaluate(self, *args):
            raise AssertionError("manual question must bypass semantic judgment")

    class Provider:
        async def stream_answer(self, question, context):
            requests.append((question, context))
            yield "回答"

    copilot = Copilot(Provider(), events.append, question_buffer=RejectBuffer(), question_gate=RejectGate())
    try:
        with pytest.raises(ValueError, match="问题内容"):
            copilot.submit_question("   ")
        for _ in range(2):
            copilot.submit_question("  React Fiber 原理與實現  ")
            assert copilot.wait_idle(2)
        assert [question for question, _ in requests] == ["React Fiber 原理与实现"] * 2
        assert requests[1][1] == [
            {"role": "user", "content": "React Fiber 原理与实现"},
            {"role": "assistant", "content": "回答"},
        ]
        assert [event.question_id for event in events if event.kind == "done"] == [1, 2]
    finally:
        copilot.close()
    with pytest.raises(RuntimeError, match="closed"):
        copilot.submit_question("主题")


def test_manual_question_invalidates_pending_judgment_even_if_cancellation_is_caught():
    judging = threading.Event()
    cancelled = threading.Event()
    requests = []

    class Gate:
        async def evaluate(self, *args):
            judging.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                return 1.0

    class Provider:
        async def stream_answer(self, question, context):
            requests.append(question)
            yield "回答"

    copilot = Copilot(Provider(), lambda event: None, question_gate=Gate(), question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("旧自动问题", 1)
        assert judging.wait(2)
        copilot.submit_question("编辑后的主题")
        assert copilot.wait_idle(2)
        assert cancelled.is_set()
        assert requests == ["编辑后的主题"]
    finally:
        copilot.close()


def test_manual_question_supersedes_running_answer_and_releases_stream():
    entered = threading.Event()
    released = threading.Event()
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            if question == "原主题":
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    released.set()
            yield "新回答"

    copilot = Copilot(Provider(), events.append)
    try:
        copilot.submit_question("原主题")
        assert entered.wait(2)
        copilot.submit_question("修正主题")
        assert copilot.wait_idle(2)
        assert released.is_set()
        assert any(event.kind == "cancelled" and event.question_id == 1 for event in events)
        assert events[-1].kind == "done"
        assert events[-1].question == "修正主题"
        assert events[-1].question_id == 2
    finally:
        copilot.close()


def test_microphone_lecture_request_reaches_answer_provider():
    requests = []
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            requests.append(question)
            yield "回答要点"

    copilot = Copilot(Provider(), events.append)
    try:
        copilot.submit_transcript("讲解一下", 1000)
        assert copilot.wait_idle(2)
        assert requests == []
        copilot.submit_transcript("react fiber原因", 4000)
        assert copilot.wait_idle(2)
        assert requests == ["讲解一下react fiber原因"]
        assert [event.kind for event in events] == ["untriggered", "start", "delta", "done"]
        assert "尚未完整" in events[0].text
    finally:
        copilot.close()


def test_submission_does_not_wait_for_provider_and_idle_tracks_queued_work():
    entered = threading.Event()
    release = threading.Event()
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            yield "第一点"
            yield "第二点"

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("问题？", 1000)
        assert not copilot.wait_idle(0)
        assert entered.wait(2)
        release.set()
        assert copilot.wait_idle(2)
        assert [e.kind for e in events] == ["start", "delta", "delta", "done"]
        assert events[-1].text == "第一点第二点"
        assert events[-1].first_token_ms >= 0
        assert events[-1].elapsed_ms >= events[-1].first_token_ms
    finally:
        release.set()
        copilot.close()


def test_superseded_answer_releases_stream_and_suppresses_stale_output():
    entered = threading.Event()
    released = threading.Event()
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            if question == "旧问题？":
                try:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        # Even a provider that catches cancellation cannot leak stale text.
                        yield "过期回答"
                finally:
                    released.set()
            else:
                yield "新回答"

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("旧问题？", 1)
        assert entered.wait(2)
        copilot.submit_transcript("新问题？", 2)
        assert copilot.wait_idle(2)
        assert released.is_set()
        assert not any(e.question_id == 1 and e.kind in {"delta", "done"} for e in events)
        assert events[-1].kind == "done"
        assert events[-1].text == "新回答"
    finally:
        copilot.close()


def test_close_cancels_running_stream_and_joins_worker():
    entered = threading.Event()
    released = threading.Event()
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "never"
            finally:
                released.set()

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal())
    copilot.submit_transcript("问题？", 1)
    assert entered.wait(2)
    copilot.close()
    copilot.close()
    assert released.is_set()
    assert not copilot._thread.is_alive()
    assert copilot.wait_idle(0)
    assert events[-1].kind == "cancelled"


def test_context_is_bounded_and_provider_errors_do_not_expose_secrets():
    contexts = []
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            contexts.append(context)
            if question == "错误？":
                raise RuntimeError("https://secret.example/?api_key=private")
            yield question + "答案"

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal(), max_context_turns=1)
    try:
        for question in ["一？", "二？", "三？", "错误？"]:
            copilot.submit_transcript(question, 1)
            assert copilot.wait_idle(2)
        assert contexts[0] == []
        assert contexts[2] == [
            {"role": "user", "content": "二？"},
            {"role": "assistant", "content": "二？答案"},
        ]
        assert events[-1].kind == "error"
        assert "private" not in events[-1].text
        assert "secret.example" not in events[-1].text
    finally:
        copilot.close()


def test_nonquestion_partial_becomes_idle_without_generation():
    events = []
    copilot = Copilot(None, events.append, question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("partial", 1, final=False)
        assert copilot.wait_idle(2)
        assert events == []
    finally:
        copilot.close()


def test_empty_answer_is_error_and_does_not_enter_context():
    contexts = []
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            contexts.append(context)
            if question == "正常？":
                yield "回答"

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("空回答？", 1)
        assert copilot.wait_idle(2)
        assert [event.kind for event in events] == ["start", "error"]
        assert "未返回回答内容" in events[-1].text
        copilot.submit_transcript("正常？", 2)
        assert copilot.wait_idle(2)
        assert contexts == [[], []]
        assert events[-1].kind == "done"
    finally:
        copilot.close()


def test_safe_provider_error_preserves_actionable_description():
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            raise AnswerProviderError("回答模型 HTTP 错误（401）")
            yield "never"

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("问题？", 1)
        assert copilot.wait_idle(2)
        assert events[-1].kind == "error"
        assert events[-1].text == "回答模型 HTTP 错误（401）"
    finally:
        copilot.close()


def test_superseded_task_emits_no_start_before_execution():
    events = []

    class Provider:
        async def stream_answer(self, question, context):
            yield "never"

    copilot = Copilot(Provider(), events.append, question_buffer=EveryFinal())
    try:
        # Directly exercise a coroutine that was constructed for an older ID.
        copilot._question_id = 2
        future = asyncio.run_coroutine_threadsafe(copilot._answer(1, "旧问题？"), copilot._loop)
        future.result(timeout=2)
        assert events == []
    finally:
        copilot.close()


def test_close_finalizes_remaining_async_generators():
    released = threading.Event()
    generators = []

    async def nested_stream():
        try:
            yield "内部数据"
        finally:
            await asyncio.sleep(0.01)
            released.set()

    class Provider:
        async def stream_answer(self, question, context):
            stream = nested_stream()
            generators.append(stream)
            await anext(stream)
            yield "回答"

    copilot = Copilot(Provider(), lambda event: None, question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("问题？", 1)
        assert copilot.wait_idle(2)
        assert not released.is_set()
    finally:
        copilot.close()
    assert released.is_set()
    assert not copilot._thread.is_alive()


def test_semantic_gate_accepts_only_high_probability_and_retains_transcript_context():
    judgments = []
    requests = []
    events = []

    class Gate:
        async def evaluate(self, question, recent):
            judgments.append((question, recent))
            return 0.96 if question == "缓存一致性展开聊聊" else 0.6

    class Provider:
        async def stream_answer(self, question, context):
            requests.append(question)
            yield "回答"

    copilot = Copilot(Provider(), events.append, question_gate=Gate())
    try:
        for index in range(8):
            copilot.submit_transcript("我们正在讨论缓存" + str(index), index)
        copilot.submit_transcript("继续聊聊缓存", 10)
        assert copilot.wait_idle(2)
        assert requests == []
        assert events[-1].kind == "skipped"
        assert len(judgments[-1][1]) == 6
        assert judgments[-1][1][0].endswith("2")
        copilot.submit_transcript("缓存一致性展开聊聊", 11)
        assert copilot.wait_idle(2)
        assert requests == ["缓存一致性展开聊聊"]
        assert events[-1].kind == "done"
        assert events[-1].decision_ms >= 0
    finally:
        copilot.close()


def test_screenshot_keyword_question_answers_without_jev_call():
    requests = []

    class Gate:
        async def evaluate(self, *args):
            raise AssertionError("keyword hits must use the fast path")

    class Provider:
        async def stream_answer(self, question, context):
            requests.append(question)
            yield "Promise 的机制"

    copilot = Copilot(Provider(), lambda _: None, question_gate=Gate())
    try:
        copilot.submit_transcript("ES6的promise有了解吗", 1)
        assert copilot.wait_idle(2)
        assert requests == ["ES6的promise有了解吗"]
    finally:
        copilot.close()


def test_unmatched_finals_use_jev_and_partials_do_not():
    judgments, requests = [], []

    class Gate:
        async def evaluate(self, text, recent):
            judgments.append((text, recent))
            return 0.99 if text == "Promise这块展开聊聊" else 0.1

    class Provider:
        async def stream_answer(self, question, context):
            requests.append(question)
            yield "回答"

    copilot = Copilot(Provider(), lambda _: None, question_gate=Gate())
    try:
        copilot.submit_transcript("Promise这块展开聊聊", 1, final=False)
        copilot.submit_transcript("今天讨论异步编程", 2)
        copilot.submit_transcript("Promise这块展开聊聊", 3)
        copilot.submit_transcript("嗯", 4)
        assert copilot.wait_idle(2)
        assert [text for text, _ in judgments] == ["今天讨论异步编程", "Promise这块展开聊聊", "嗯"]
        assert judgments[1][1] == ["今天讨论异步编程"]
        assert requests == ["Promise这块展开聊聊"]
    finally:
        copilot.close()


def test_duplicate_semantic_requests_are_judged_but_not_answered_twice():
    requests, judgments = [], []

    class Gate:
        async def evaluate(self, text, recent):
            judgments.append(text)
            return 0.99

    class Provider:
        async def stream_answer(self, question, context):
            requests.append(question)
            yield "回答"

    copilot = Copilot(Provider(), lambda _: None, question_gate=Gate())
    try:
        for timestamp in (1, 2):
            copilot.submit_transcript("Promise这块展开聊聊", timestamp)
            assert copilot.wait_idle(2)
        assert len(judgments) == 2
        assert requests == ["Promise这块展开聊聊"]
    finally:
        copilot.close()


def test_screenshot_meta_discussion_reaches_semantic_judge_but_not_answer_provider():
    judgments = []
    events = []

    class Gate:
        async def evaluate(self, text, recent):
            judgments.append(text)
            return 0.05

    text = "就是说很明确的就是请问或者说请说出啊什么东西的"
    copilot = Copilot(None, events.append, question_gate=Gate())
    try:
        copilot.submit_transcript(text, 1)
        assert copilot.wait_idle(2)
        assert judgments == [text]
        assert [event.kind for event in events] == ["checking", "skipped"]
    finally:
        copilot.close()


def test_failed_judgment_does_not_fall_back_to_keyword_answer():
    events = []

    class Gate:
        async def evaluate(self, question, recent):
            raise QuestionGateError("语义判断 HTTP 错误（401）")

    copilot = Copilot(None, events.append, question_gate=Gate())
    try:
        copilot.submit_transcript("缓存一致性展开聊聊", 1)
        assert copilot.wait_idle(2)
        assert [event.kind for event in events] == ["checking", "gate_error"]
        assert "401" in events[-1].text
    finally:
        copilot.close()


def test_fallback_judgments_are_ordered_and_not_cancelled_by_next_sentence():
    entered = threading.Event()
    release = threading.Event()
    questions = []
    judgments = []

    class Gate:
        async def evaluate(self, question, recent):
            judgments.append(question)
            if question == "旧问题？":
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.005)
                return 0.1
            return 0.99

    class Provider:
        async def stream_answer(self, question, context):
            questions.append(question)
            yield "回答"

    copilot = Copilot(Provider(), lambda event: None, question_gate=Gate(), question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("旧问题？", 1)
        assert entered.wait(2)
        copilot.submit_transcript("新问题？", 2)
        release.set()
        assert copilot.wait_idle(2)
        assert judgments == ["旧问题？", "新问题？"]
        assert questions == ["新问题？"]
    finally:
        release.set()
        copilot.close()


def test_rejected_candidate_does_not_interrupt_current_answer():
    entered = threading.Event()
    release = threading.Event()
    events = []

    class Gate:
        async def evaluate(self, question, recent):
            return 0.99 if question == "真实问题？" else 0.1

    class Provider:
        async def stream_answer(self, question, context):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            yield "真实回答"

    copilot = Copilot(Provider(), events.append, question_gate=Gate(), question_buffer=EveryFinal())
    try:
        copilot.submit_transcript("真实问题？", 1)
        assert entered.wait(2)
        copilot.submit_transcript("普通讨论？", 2)
        release.set()
        assert copilot.wait_idle(2)
        assert any(event.kind == "done" and event.text == "真实回答" for event in events)
        assert not any(event.kind == "cancelled" for event in events)
    finally:
        release.set()
        copilot.close()
