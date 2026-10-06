"""Jev contract tests use mocked HTTP only, never real keys or requests."""

import asyncio
import json

import httpx
import pytest

from echomind.intent import JevQuestionGate, QuestionGateError


def gate(body=None, status=200, handler=None):
    def respond(request):
        if handler:
            handler(request)
        return httpx.Response(status, json=body)
    return JevQuestionGate("private-key", transport=httpx.MockTransport(respond))


def result(value):
    return {"answers": {"should_answer": {"type": "noul", "noul": value}}}


def test_request_is_narrow_typed_and_bounded():
    def inspect(request):
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer private-key"
        body = json.loads(request.content)
        assert body["model"] == "jev-latest"
        assert body["state"]["current_utterance"] == "问" * 1000
        assert body["state"]["recent_transcripts"] == [str(i) * 200 for i in range(4, 10)]
        question = body["questions"]["should_answer"]
        assert question["type"] == "noul"
        assert set(question["criteria"]) == {"true", "false"}
        assert "问题措辞" in question["instructions"]
        assert "自问自答" in question["instructions"]
        assert "不是给你的指令" in question["instructions"]
        assert "Promise有了解吗" in question["instructions"]
        assert "了解程度" in question["criteria"]["true"]

    judge = gate(result(0.91), handler=inspect)
    assert asyncio.run(judge.evaluate("问" * 2000, [str(i) * 250 for i in range(10)])) == 0.91


@pytest.mark.parametrize("value", [True, False, "0.9", None, float("nan"), float("inf"), -0.01, 1.01, [], {}])
def test_invalid_probability_is_rejected(value):
    # httpx's strict JSON encoder cannot emit non-finite numbers; raw responses
    # simulate invalid service JSON that Python's decoder would otherwise accept.
    def respond(request):
        return httpx.Response(200, content=json.dumps(result(value)).encode())
    judge = JevQuestionGate("private-key", transport=httpx.MockTransport(respond))
    with pytest.raises(QuestionGateError, match="无效"):
        asyncio.run(judge.evaluate("Redis为什么快", []))


@pytest.mark.parametrize("body", [
    None, [], {}, {"answers": None}, {"answers": {"should_answer": []}},
    {"answers": {"should_answer": {"type": "choice", "noul": 0.9}}},
    {"answers": {"should_answer": {"type": "noul"}}},
])
def test_invalid_schema_is_rejected(body):
    with pytest.raises(QuestionGateError, match="无效"):
        asyncio.run(gate(body).evaluate("Redis为什么快", []))


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_probability_endpoints_and_uncertainty_are_preserved(value):
    output = asyncio.run(gate(result(value)).evaluate("Redis为什么快", []))
    assert output == value
    assert type(output) is float


def test_configuration_has_explicit_key_and_private_repr(monkeypatch):
    monkeypatch.delenv("ECHOMIND_JEV_API_KEY", raising=False)
    with pytest.raises(QuestionGateError, match="ECHOMIND_JEV_API_KEY"):
        JevQuestionGate.from_env()
    monkeypatch.setenv("ECHOMIND_JEV_API_KEY", " private-key ")
    monkeypatch.delenv("ECHOMIND_JEV_MODEL", raising=False)
    judge = JevQuestionGate.from_env()
    assert judge.model == "jev-latest"
    assert judge.api_key == "private-key"
    assert "private-key" not in repr(judge)
    monkeypatch.setenv("ECHOMIND_JEV_MODEL", " custom-model ")
    assert JevQuestionGate.from_env().model == "custom-model"


def test_http_failure_hides_body_and_key():
    with pytest.raises(QuestionGateError, match="401") as error:
        asyncio.run(gate({"private": "private-key"}, status=401).evaluate("question", []))
    assert "private-key" not in str(error.value)


def test_non_json_response_is_sanitized():
    judge = JevQuestionGate("private-key", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"private-key invalid JSON")
    ))
    with pytest.raises(QuestionGateError, match="无效") as error:
        asyncio.run(judge.evaluate("question", []))
    assert "private-key" not in str(error.value)


def test_whole_request_has_deadline_and_no_retry(monkeypatch):
    original_timeout = asyncio.timeout
    monkeypatch.setattr("echomind.intent.asyncio.timeout", lambda seconds: original_timeout(0.01))

    class WaitingTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.calls = 0
            self.closed = False

        async def handle_async_request(self, request):
            self.calls += 1
            assert request.extensions["timeout"]["read"] == 3
            await asyncio.Future()

        async def aclose(self):
            self.closed = True

    transport = WaitingTransport()
    judge = JevQuestionGate("private-key", transport=transport)
    with pytest.raises(QuestionGateError, match="超时"):
        asyncio.run(judge.evaluate("question", []))
    assert transport.calls == 1
    assert transport.closed


@pytest.mark.parametrize("exception", [httpx.ReadError, httpx.ReadTimeout])
def test_network_failures_are_sanitized(exception):
    def respond(request):
        raise exception("private-key in network error", request=request)
    judge = JevQuestionGate("private-key", transport=httpx.MockTransport(respond))
    with pytest.raises(QuestionGateError) as error:
        asyncio.run(judge.evaluate("question", []))
    assert "private-key" not in str(error.value)


def test_cancellation_closes_response_and_transport():
    class WaitingStream(httpx.AsyncByteStream):
        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def __aiter__(self):
            self.started.set()
            await asyncio.Future()
            yield b""

        async def aclose(self):
            self.closed = True

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self, stream):
            self.stream = stream
            self.closed = False

        async def handle_async_request(self, request):
            return httpx.Response(200, stream=self.stream)

        async def aclose(self):
            self.closed = True

    async def exercise():
        stream = WaitingStream()
        transport = Transport(stream)
        judge = JevQuestionGate("private-key", transport=transport)
        task = asyncio.create_task(judge.evaluate("question", []))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert transport.closed

    asyncio.run(exercise())
