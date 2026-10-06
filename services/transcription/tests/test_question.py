"""Question detection tests independent of ASR models and remote services."""

import pytest

from echomind.question import PausedQuestion, Question, QuestionBuffer


@pytest.mark.parametrize("text", [
    "Redis为什么这么快", "怎么排查这个问题", "如何设计缓存", "什么是事务",
    "Redis是什么", "请介绍Redis", "请你解释事务", "谈谈缓存一致性", "解释一下索引",
    "讲解一下React Fiber原理", "讲解一下React Vibre原理", "讲解一下react library 为例",
    "请讲解React Fiber原理", "请你讲解React Fiber原理", "五乘六等于多少", "桑乘桑等于多少",
    "1加1等于几。", "一加一等于几", "请问五乘六等于几呢", "2.5乘以3等于几",
])
def test_explicit_questions_without_punctuation(text: str) -> None:
    assert QuestionBuffer().push(text, 1234) == Question(text, 1234)


@pytest.mark.parametrize("text", [
    "今天讨论Redis", "我不知道为什么这么慢", "我不清楚如何使用Redis",
    "他说Redis为什么这么快", "她问怎么使用缓存", "“Redis为什么这么快”",
    '他说"如何设计缓存"', "我明白了为什么需要索引",
    "我来讲解一下React Fiber原理", "我给你讲解一下React Fiber原理",
    "我们正在讲解一下React Fiber原理", "他说讲解一下React Fiber原理", "我不知道五乘六等于多少",
    "我有几个方案", "他说一加一等于几", "一加一等于二", "1加1等于进。",
])
def test_statements_and_reported_questions_are_ignored(text: str) -> None:
    assert QuestionBuffer().push(text, 1000) is None


def test_partials_never_trigger_or_accumulate() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("Redis为什么", 100, final=False) is None
    assert buffer.push("Redis为什么这么快", 200, final=False) is None
    assert buffer.push("今天讨论Redis", 300) is None
    assert buffer.push("如何设计缓存", 400) == Question("如何设计缓存", 400)


def test_incomplete_final_carries_to_next_final() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("Redis为什么", 1000) is None
    assert buffer.push("这么快", 2000) == Question("Redis为什么这么快", 2000)
    assert buffer.push("这是回答", 3000) is None


def test_partial_revision_does_not_change_pending_final() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("Redis为什么", 1000) is None
    assert buffer.push("Redis为什么这么慢", 1500, final=False) is None
    assert buffer.push("这么快", 2000) == Question("Redis为什么这么快", 2000)


def test_repeated_final_fragment_is_not_concatenated_twice() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("Redis为什么", 1000) is None
    assert buffer.push("Redis为什么这么快", 2000) == Question("Redis为什么这么快", 2000)


def test_incomplete_request_and_what_carry() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("请介绍", 1000) is None
    assert buffer.push("Redis", 2000) == Question("请介绍Redis", 2000)
    assert buffer.push("什么", 3000) is None
    assert buffer.push("是事务", 4000) == Question("什么是事务", 4000)


def test_split_lecture_request_from_microphone_screenshot() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("讲解一下", 1000) is None
    assert buffer.push("react fiber原因", 4000) == Question("讲解一下react fiber原因", 4000)


def test_traditional_screenshot_is_detected_and_simplified() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("解釋一下React-Fibre原理", 1000) == Question("解释一下React-Fibre原理", 1000)
    assert buffer.push("解释一下React-Fibre原理", 2000) is None
    assert "重复" in buffer.decision


def test_stale_fragment_expires_and_context_is_bounded() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("Redis为什么", 1000) is None
    assert buffer.push("这么快", 10_000) is None
    assert buffer.push("背景" * 200 + "Redis为什么", 11_000) is None
    question = buffer.push("这么快", 12_000)
    assert question is not None
    assert len(question.text) <= 200


def test_duplicate_ignores_punctuation_and_expires() -> None:
    buffer = QuestionBuffer()
    assert buffer.push("Redis 为什么这么快？", 1000) is not None
    assert buffer.push("Redis为什么这么快", 2000) is None
    assert buffer.push("Redis为什么这么快", 32_000) is not None


def test_direct_question_after_reported_sentence() -> None:
    assert QuestionBuffer().push("他说Redis为什么这么快。如何设计缓存", 1000) is not None


@pytest.mark.parametrize("text", [
    "就是说很明确的就是请问或者说请说出啊什么东西的",
    "我们讨论问题的表达方式为什么重要",
    "普通对话会被识别成问题什么的",
    "我来解释为什么Redis这么快",
    "原因就是Redis为什么可以这么快",
    "之所以这么做是因为我知道怎么实现",
])
def test_meta_discussion_and_self_explanation_do_not_trigger(text):
    assert QuestionBuffer().push(text, 1000) is None


@pytest.mark.parametrize("text", ["能不能介绍缓存一致性", "你了解Redis吗", "这个方案是否可行", "是不是要加锁",
                                 "Redis快的原因是什么", "为什么普通对话会被识别成问题"])
def test_more_direct_question_forms(text):
    assert QuestionBuffer().push(text, 1000) is not None


def test_semantic_candidates_are_not_vetoed_by_reporting_keywords():
    text = "请告诉我为什么普通对话会被识别成问题"
    assert QuestionBuffer(semantic_candidates=True).push(text, 1000) is not None
    discussion = "就是说很明确的就是请问或者说请说出啊什么东西的"
    assert QuestionBuffer(semantic_candidates=True).push(discussion, 1000) is not None
    assert QuestionBuffer().push(discussion, 1000) is None


@pytest.mark.parametrize("text", [
    "设计一个AI前端灰度发布方案", "请设计一种发布方案", "帮我实现一个缓存",
    "制定一套发布计划", "给出一个方案",
])
def test_task_requests_are_questions(text):
    assert QuestionBuffer().push(text, 1000) == Question(text, 1000)


@pytest.mark.parametrize("text", ["我设计一个方案", "他说设计一个方案", "设计模式很重要", "支持按用户ID放量"])
def test_statements_are_not_task_requests(text):
    assert QuestionBuffer().push(text, 1000) is None


def test_paused_task_keeps_constraints_and_suspect_flag():
    buffer = PausedQuestion()
    first = "设计一个AI前端灰度发布方案。"
    second = "支持按用户ID设备类型地理位置逐步放量新功能"
    assert buffer.push(first, 1000, True) == (first, True, False)
    assert buffer.push(second, 8000) == (first[:-1] + "，" + second, True, True)


def test_paused_garbled_scheme_preserves_raw_text():
    buffer = PausedQuestion()
    buffer.push("这几个AI前端挥布发布方案", 1000)
    assert buffer.push("支持用户按ID设备类型地理位置逐步放量新功能", 8000) == (
        "这几个AI前端挥布发布方案，支持用户按ID设备类型地理位置逐步放量新功能", False, True,
    )


def test_paused_incomplete_opening_and_repeated_revision():
    buffer = PausedQuestion()
    buffer.push("讲解一下", 1000)
    assert buffer.push("讲解一下React Fiber原理", 2000) == ("讲解一下React Fiber原理", False, True)


def test_paused_context_expires_even_after_filler():
    buffer = PausedQuestion()
    buffer.push("设计一个发布方案", 1000)
    assert buffer.push("OK", 8000) == ("OK", False, False)
    assert buffer.push("支持按用户ID放量", 13_001) == ("支持按用户ID放量", False, False)


def test_paused_filler_does_not_become_question_text():
    buffer = PausedQuestion()
    buffer.push("设计一个发布方案", 1000)
    buffer.push("嗯", 2000)
    assert buffer.push("支持按用户ID放量", 3000)[0] == "设计一个发布方案，支持按用户ID放量"


def test_unrelated_statement_clears_paused_context():
    buffer = PausedQuestion()
    buffer.push("设计一个发布方案", 1000)
    buffer.push("我们现在去吃饭", 2000)
    assert not buffer.push("支持按用户ID放量", 3000)[2]


def test_paused_join_is_bounded_by_parts_and_length():
    buffer = PausedQuestion()
    buffer.push("设计一个发布方案", 1000)
    assert buffer.push("支持按用户ID放量", 2000)[2]
    assert buffer.push("并且可回滚", 3000)[2]
    assert not buffer.push("要求支持灰度", 4000)[2]
    buffer.push("设计一个" + "a" * 190, 5000)
    assert not buffer.push("支持按用户ID放量", 6000)[2]


def test_explicit_clear_prevents_paused_join():
    buffer = PausedQuestion()
    buffer.push("设计一个发布方案", 1000)
    buffer.clear()
    assert not buffer.push("支持按用户ID放量", 2000)[2]
