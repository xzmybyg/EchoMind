"""Uncertain transcripts never silently change the user's question."""

import asyncio
import threading

from echomind.copilot import Copilot
from echomind.repair import RepairResult


class Provider:
    def __init__(self):
        self.requests = []

    async def stream_answer(self, question, context):
        self.requests.append(question)
        yield "回答"


def test_local_arithmetic_suggestion_requires_explicit_submission():
    provider, events = Provider(), []
    copilot = Copilot(provider, events.append)
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert copilot.wait_idle(2)
        assert provider.requests == []
        assert [event.kind for event in events] == ["reviewing", "needs_confirmation"]
        assert events[-1].question == "1加1等于进。"
        assert events[-1].text
        copilot.submit_question("1加1等于几？")
        assert copilot.wait_idle(2)
        assert provider.requests == ["1加1等于几？"]
        assert events[-1].kind == "done"
    finally:
        copilot.close()


def test_clear_question_does_not_call_reviser():
    class Reviser:
        async def review(self, *args):
            raise AssertionError("clear questions do not need remote repair")

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("请解释一下Redis为什么快", 1)
        assert copilot.wait_idle(2)
        assert provider.requests == ["请解释一下Redis为什么快"]
        assert not any(event.kind == "reviewing" for event in events)
    finally:
        copilot.close()


def test_suspect_flag_only_reviews_explicit_request_and_not_partial():
    reviewed = []

    class Reviser:
        async def review(self, text, recent):
            reviewed.append(text)
            return RepairResult(text, text, False, "")

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("我们正在讨论React", 1, suspect=True)
        copilot.submit_transcript("解释一下React", 2, final=False, suspect=True)
        assert copilot.wait_idle(2)
        assert reviewed == []
        copilot.submit_transcript("解释一下React原理", 3, suspect=True)
        assert copilot.wait_idle(2)
        assert reviewed == ["解释一下React原理"]
        assert provider.requests == ["解释一下React原理"]
    finally:
        copilot.close()


def test_keyword_question_reviews_without_second_semantic_call():
    judged = []

    class Reviser:
        async def review(self, text, recent):
            return RepairResult(text, text, False, "")

    class Gate:
        async def evaluate(self, text, recent):
            judged.append((text, recent))
            return 0.1

    provider = Provider()
    copilot = Copilot(provider, lambda _: None, question_gate=Gate(), question_reviser=Reviser())
    try:
        copilot.submit_transcript("讲解一下", 1)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("Redis为什么快", 2, suspect=True)
        assert copilot.wait_idle(2)
        assert provider.requests == ["讲解一下Redis为什么快"]
        assert judged == [("讲解一下", [])]
    finally:
        copilot.close()


def test_repair_duplicate_is_bounded_and_no_second_remote_call():
    reviewed = []

    class Reviser:
        async def review(self, text, recent):
            reviewed.append(text)
            return RepairResult(text, "1加1等于几？", True, "疑似误识别")

    events = []
    copilot = Copilot(Provider(), events.append, question_reviser=Reviser())
    try:
        for _ in range(2):
            copilot.submit_transcript("1加1等于进。", 1)
            assert copilot.wait_idle(2)
        assert reviewed == ["1加1等于进。"]
        assert len([event for event in events if event.kind == "needs_confirmation"]) == 1
        assert copilot._reviewed.maxlen == 16
    finally:
        copilot.close()


def test_repair_failure_is_fail_closed_and_does_not_expose_remote_exception():
    class Reviser:
        async def review(self, *args):
            raise RuntimeError("https://private.example/key=secret")

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert copilot.wait_idle(2)
        assert provider.requests == []
        assert events[-1].kind == "repair_error"
        assert "secret" not in str(events)
        assert "private.example" not in str(events)
    finally:
        copilot.close()


def test_filler_does_not_cancel_or_replace_reviewing_status():
    entered, release = threading.Event(), threading.Event()

    class Reviser:
        async def review(self, text, recent):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            return RepairResult(text, "1加1等于几？", True, "疑似误识别")

    events = []
    copilot = Copilot(Provider(), events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert entered.wait(2)
        copilot.submit_transcript("嗯", 2, suspect=True)
        release.set()
        assert copilot.wait_idle(2)
        assert [event.kind for event in events] == ["reviewing", "needs_confirmation"]
    finally:
        release.set()
        copilot.close()


def test_manual_submission_suppresses_even_cancellation_ignoring_reviser():
    entered, cancelled = threading.Event(), threading.Event()

    class Reviser:
        async def review(self, text, recent):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                return RepairResult(text, "错误的旧建议", True, "")

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert entered.wait(2)
        copilot.submit_question("用户确认的问题")
        assert copilot.wait_idle(2)
        assert cancelled.is_set()
        assert provider.requests == ["用户确认的问题"]
        assert not any(event.kind == "needs_confirmation" for event in events)
    finally:
        copilot.close()


def test_clear_question_supersedes_review():
    entered = threading.Event()

    class Reviser:
        async def review(self, text, recent):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return RepairResult(text, "旧建议", True, "")

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert entered.wait(2)
        copilot.submit_transcript("Redis为什么快", 2)
        assert copilot.wait_idle(2)
        assert provider.requests == ["Redis为什么快"]
        assert not any(event.kind == "needs_confirmation" for event in events)
    finally:
        copilot.close()


def test_new_suspect_supersedes_old_review():
    entered = threading.Event()

    class Reviser:
        async def review(self, text, recent):
            if "进" in text:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            return RepairResult(text, text, True, "确认")

    events = []
    copilot = Copilot(Provider(), events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert entered.wait(2)
        copilot.submit_transcript("解释一下React Vibre原理", 2)
        assert copilot.wait_idle(2)
        suggestions = [event for event in events if event.kind == "needs_confirmation"]
        assert len(suggestions) == 1
        assert suggestions[0].question == "解释一下React Vibre原理"
    finally:
        copilot.close()


def test_close_cancels_review_without_late_confirmation():
    entered, released = threading.Event(), threading.Event()

    class Reviser:
        async def review(self, text, recent):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

    events = []
    copilot = Copilot(Provider(), events.append, question_reviser=Reviser())
    copilot.submit_transcript("1加1等于进。", 1)
    assert entered.wait(2)
    copilot.close()
    assert released.is_set()
    assert copilot.wait_idle(0)
    assert not copilot._thread.is_alive()
    assert not any(event.kind == "needs_confirmation" for event in events)


def test_semantic_change_is_confirmed_even_if_reviser_marks_it_safe():
    class Reviser:
        async def review(self, text, recent):
            return RepairResult(text, "2加2等于几？", False, "")

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_reviser=Reviser())
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert copilot.wait_idle(2)
        assert provider.requests == []
        assert events[-1].kind == "needs_confirmation"
        assert events[-1].text == "2加2等于几？"
    finally:
        copilot.close()


def test_duplicate_review_expiration_allows_new_attempt():
    events = []
    copilot = Copilot(Provider(), events.append)
    try:
        copilot.submit_transcript("1加1等于进。", 1)
        assert copilot.wait_idle(2)
        key, when = copilot._reviewed[0]
        copilot._reviewed[0] = (key, when - 31)
        copilot.submit_transcript("1加1等于进。", 2)
        assert copilot.wait_idle(2)
        assert len([event for event in events if event.kind == "needs_confirmation"]) == 2
    finally:
        copilot.close()


def test_review_discards_previous_incomplete_request_before_future_transcript():
    provider = Provider()
    copilot = Copilot(provider, lambda _: None)
    try:
        copilot.submit_transcript("讲解一下", 1)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("1加1等于进。", 2)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("Redis缓存", 3)
        assert copilot.wait_idle(2)
        assert provider.requests == []
    finally:
        copilot.close()


def test_offline_low_confidence_needs_confirmation_without_guessing():
    provider, events = Provider(), []
    copilot = Copilot(provider, events.append)
    try:
        copilot.submit_transcript("解释一下React原理", 1, suspect=True)
        assert copilot.wait_idle(2)
        assert provider.requests == []
        assert events[-1].kind == "needs_confirmation"
        assert events[-1].text == "解释一下React原理"
        assert "置信度偏低" in events[-1].reason
    finally:
        copilot.close()
