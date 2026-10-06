"""Review suggestions are bounded, private and never auto-applied."""

import asyncio
import json

import httpx
import pytest

from echomind.repair import (
    OpenAICompatibleQuestionReviser, local_review, normalize_question, suspect_question,
)


@pytest.mark.parametrize("text", ["1加1等于进。", "解释一下 React Vibre 原理", "请讲解 React Fibre 原理", "解释一下React Vibre原理", "解释一下ReactFibre原理"])
def test_suspect_limited_to_malformed_arithmetic_and_requested_terms(text):
    assert suspect_question(text)


@pytest.mark.parametrize("text", ["1加1等于几。", "一加一等于多少", "1加1等于2", "我在讲解 React Vibre 原理", "React Fibre很好用", "你今天还好吗", "我不知道1加1等于进", "解释一下React Vibreation原理", "解释一下React librarytools原理"])
def test_normal_chatter_not_sent_for_review(text):
    assert not suspect_question(text)


def test_local_arithmetic_suggestion_requires_confirmation_and_preserves_operands():
    result = local_review("13加27等于进。")
    assert result.corrected == "13加27等于几？"
    assert result.original == "13加27等于进。"
    assert result.needs_confirmation
    assert local_review("解释一下 React Vibre 原理").needs_confirmation
    assert not local_review("1加1等于几").needs_confirmation
    assert normalize_question("  1加1\n等于几  ") == "1加1 等于几"


def reviewer(content, inspect=None, *, status=200, finish_reason="stop"):
    def respond(request):
        if inspect:
            inspect(request)
        return httpx.Response(status, json={"choices": [{"finish_reason": finish_reason,
              "message": {"content": content if isinstance(content, str) else json.dumps(content)}}]})
    return OpenAICompatibleQuestionReviser("https://example.invalid/v1", "model", "hidden-key",
                                            transport=httpx.MockTransport(respond))


def test_json_mode_bounds_inputs_and_never_prints_key():
    text = "解释一下 React Vibre 原理"
    def inspect(request):
        assert str(request.url) == "https://example.invalid/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer hidden-key"
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["response_format"] == {"type": "json_object"}
        system = body["messages"][0]["content"]
        assert "不回答" in system and "不进行算术求值" in system
        state = json.loads(body["messages"][1]["content"])
        assert state["current"] == text
        assert len(state["recent"]) == 4
        assert all(len(item) <= 200 for item in state["recent"])
    model = reviewer({"status": "suggestion", "corrected": "解释一下 React Fiber 原理"}, inspect)
    result = asyncio.run(model.review(text, ["x" * 300] * 20))
    assert result.needs_confirmation
    assert result.corrected == "解释一下 React Fiber 原理"
    assert "hidden-key" not in repr(model)


def test_model_cannot_declare_a_changed_number_safe():
    result = asyncio.run(reviewer({"status": "unchanged", "corrected": "1加2等于几"}).review("1加1等于几", []))
    assert result.needs_confirmation


def test_valid_unchanged_question_can_proceed():
    text = "解释一下 React Fiber 原理"
    result = asyncio.run(reviewer({"status": "unchanged", "corrected": text}).review(text, []))
    assert not result.needs_confirmation
    assert result.corrected == text


@pytest.mark.parametrize("data", ["not json", {"status": "unchanged"}, {"status": "unknown", "corrected": "x"}, {"status": "unchanged", "corrected": 2}, {"status": "unchanged", "corrected": ""}, {"status": "uncertain", "corrected": "random guess"}, {"status": "suggestion", "corrected": "x" * 501}, []])
def test_invalid_or_uncertain_response_keeps_original(data):
    result = asyncio.run(reviewer(data).review("1加1等于进", []))
    assert result.needs_confirmation
    assert result.corrected == result.original == "1加1等于进"


def test_truncated_response_not_accepted():
    result = asyncio.run(reviewer({"status": "unchanged", "corrected": "1加1等于几"}, finish_reason="length").review("1加1等于几", []))
    assert result.needs_confirmation


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_never_expose_body_or_credentials(status):
    result = asyncio.run(reviewer("hidden-key/private-text", status=status).review("1加1等于进", []))
    assert result.needs_confirmation
    assert str(status) in result.reason
    assert "hidden-key" not in result.reason and "private-text" not in result.reason


def test_transport_failure_keeps_original_and_redacts_detail():
    def fail(request):
        raise httpx.ConnectError("hidden-key/private-text")
    model = OpenAICompatibleQuestionReviser("https://example.invalid", "model", "hidden-key", httpx.MockTransport(fail))
    result = asyncio.run(model.review("1加1等于进", []))
    assert result.needs_confirmation
    assert result.corrected == result.original
    assert "hidden-key" not in result.reason


def test_long_input_and_invalid_address_do_not_request():
    def never(request):
        pytest.fail("should not send request")
    model = OpenAICompatibleQuestionReviser("https://user:hidden-key@example.invalid", "model", "hidden-key", httpx.MockTransport(never))
    assert asyncio.run(model.review("x" * 501, [])).needs_confirmation
    assert asyncio.run(model.review("1加1等于进", [])).needs_confirmation


def test_cancellation_propagates_instead_of_becoming_confirmation():
    async def pause(request):
        raise asyncio.CancelledError
    model = OpenAICompatibleQuestionReviser("https://example.invalid", "model", "hidden-key", httpx.MockTransport(pause))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(model.review("1加1等于进", []))
