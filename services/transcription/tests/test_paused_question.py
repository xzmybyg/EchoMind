"""Pause handling uses raw fragments, never silently invents a lost request."""

from echomind.copilot import Copilot
from echomind.repair import RepairResult


class Provider:
    def __init__(self):
        self.requests = []

    async def stream_answer(self, question, context):
        self.requests.append(question)
        yield "回答"


def test_task_constraints_update_answer_without_semantic_roundtrip():
    class Gate:
        async def evaluate(self, *args):
            raise AssertionError("clear tasks take the local fast path")

    provider = Provider()
    copilot = Copilot(provider, lambda _: None, question_gate=Gate())
    try:
        copilot.submit_transcript("设计一个AI前端灰度发布方案", 1000)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("支持按用户ID设备类型地理位置逐步放量新功能", 8000)
        assert copilot.wait_idle(2)
        assert provider.requests == [
            "设计一个AI前端灰度发布方案",
            "设计一个AI前端灰度发布方案，支持按用户ID设备类型地理位置逐步放量新功能",
        ]
    finally:
        copilot.close()


def test_garbled_screenshot_is_reviewed_with_all_constraints_then_confirmed():
    judged, reviewed, events = [], [], []
    original = "这几个AI前端挥布发布方案"
    constraints = "支持用户按ID设备类型地理位置逐步放量新功能"
    corrected = "设计一个AI前端灰度发布方案，支持按用户ID设备类型地理位置逐步放量新功能"

    class Gate:
        async def evaluate(self, text, recent):
            judged.append(text)
            return 0.03

    class Reviser:
        async def review(self, text, recent):
            reviewed.append((text, recent))
            return RepairResult(text, corrected, True, "核对疑似误听的任务开头")

    provider = Provider()
    copilot = Copilot(provider, events.append, question_gate=Gate(), question_reviser=Reviser())
    try:
        copilot.submit_transcript(original, 1000)
        assert copilot.wait_idle(2)
        copilot.submit_transcript(constraints, 8000)
        assert copilot.wait_idle(2)
        assert judged == [original]
        assert reviewed == [(original + "，" + constraints, [original])]
        assert provider.requests == []
        assert events[-1].kind == "needs_confirmation"
        assert events[-1].question == original + "，" + constraints
        assert events[-1].text == corrected
        copilot.submit_question(corrected)
        assert copilot.wait_idle(2)
        assert provider.requests == [corrected]
    finally:
        copilot.close()


def test_unmatched_scheme_and_constraints_reach_semantic_judge_together():
    judged = []

    class Gate:
        async def evaluate(self, text, recent):
            judged.append(text)
            return 0.98 if "，支持" in text else 0.1

    provider = Provider()
    copilot = Copilot(provider, lambda _: None, question_gate=Gate())
    try:
        copilot.submit_transcript("AI前端灰度发布方案", 1000)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("支持按用户ID放量", 8000)
        assert copilot.wait_idle(2)
        assert judged == ["AI前端灰度发布方案", "AI前端灰度发布方案，支持按用户ID放量"]
        assert provider.requests == [judged[-1]]
    finally:
        copilot.close()


def test_offline_garbled_scheme_is_not_silently_answered():
    provider, events = Provider(), []
    copilot = Copilot(provider, events.append)
    try:
        copilot.submit_transcript("这几个AI前端挥布发布方案", 1000)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("支持按用户ID放量", 8000)
        assert copilot.wait_idle(2)
        assert provider.requests == []
        assert events[-1].kind == "needs_confirmation"
    finally:
        copilot.close()


def test_manual_question_clears_automatic_pause_context():
    provider = Provider()
    copilot = Copilot(provider, lambda _: None)
    try:
        copilot.submit_transcript("设计一个发布方案", 1000)
        assert copilot.wait_idle(2)
        copilot.submit_question("用户的新问题")
        assert copilot.wait_idle(2)
        copilot.submit_transcript("支持按用户ID放量", 8000)
        assert copilot.wait_idle(2)
        assert provider.requests == ["设计一个发布方案", "用户的新问题"]
    finally:
        copilot.close()


def test_skipped_status_identifies_utterance_not_previous_question():
    class Gate:
        async def evaluate(self, *args):
            return 0.03

    provider, events = Provider(), []
    copilot = Copilot(provider, events.append, question_gate=Gate())
    try:
        copilot.submit_transcript("设计一个发布方案", 1000)
        assert copilot.wait_idle(2)
        copilot.submit_transcript("OK", 2000)
        assert copilot.wait_idle(2)
        assert events[-1].kind == "skipped"
        assert "“OK”" in events[-1].text
        assert provider.requests == ["设计一个发布方案"]
    finally:
        copilot.close()


def test_filler_does_not_complete_incomplete_task():
    provider = Provider()
    copilot = Copilot(provider, lambda _: None)
    try:
        copilot.submit_transcript("设计一个", 1000)
        copilot.submit_transcript("嗯", 2000)
        assert copilot.wait_idle(2)
        assert provider.requests == []
        copilot.submit_transcript("灰度发布方案", 10_000)
        assert copilot.wait_idle(2)
        assert provider.requests == ["设计一个灰度发布方案"]
    finally:
        copilot.close()


def test_manual_question_clears_incomplete_buffer_too():
    provider = Provider()
    copilot = Copilot(provider, lambda _: None)
    try:
        copilot.submit_transcript("讲解一下", 1000)
        assert copilot.wait_idle(2)
        copilot.submit_question("用户确认的问题")
        assert copilot.wait_idle(2)
        copilot.submit_transcript("我们在吃饭", 2000)
        assert copilot.wait_idle(2)
        assert provider.requests == ["用户确认的问题"]
    finally:
        copilot.close()
