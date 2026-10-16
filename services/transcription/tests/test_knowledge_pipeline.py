import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from pathlib import Path
from unittest.mock import patch

from echomind.copilot import Copilot
from echomind.knowledge import KnowledgeHit
from echomind.llm import OpenAICompatibleProvider
from echomind.desktop_session import DesktopSession, SessionConfig


HIT = KnowledgeHit(1, 'doc1', 'Redis', '面试题.md', '第 2 行', 'Redis 在内存中读写。', .9)


def test_only_selected_material_is_passed_as_user_data_and_streams():
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n')
    provider = OpenAICompatibleProvider('https://example.invalid', 'model', 'test', transport=httpx.MockTransport(respond))
    async def collect():
        return [p async for p in provider.stream_answer('Redis 原理', [], knowledge=(HIT,))]
    assert asyncio.run(collect()) == ['answer']
    body = requests[0]
    assert body['stream']
    assert [m['role'] for m in body['messages']] == ['system', 'user']
    assert '不可信' in body['messages'][0]['content']
    assert '不是候选人的个人经历' in body['messages'][0]['content']
    data = json.loads(body['messages'][-1]['content'])
    assert data['本地参考资料'] == [{'编号':'K1','标题':'Redis','来源':'面试题.md','位置':'第 2 行','内容':HIT.text}]
    assert 'document_id' not in json.dumps(data) and 'origin' not in json.dumps(data)


def test_retrieval_sources_streaming_and_followup_history():
    queries, calls, events = [], [], []
    class Knowledge:
        notice = '混合检索'
        def retrieve(self, query, *, cancel):
            queries.append(query)
            return [HIT]
    class Provider:
        async def stream_answer(self, question, context, *, knowledge=()):
            calls.append((question, context, knowledge))
            yield '第一段'
            yield '第二段'
    copilot = Copilot(Provider(), events.append, knowledge_service=Knowledge())
    try:
        copilot.submit_question('Redis 原理')
        assert copilot.wait_idle(3)
        copilot.submit_question('详细一点')
        assert copilot.wait_idle(3)
        assert queries == ['Redis 原理', 'Redis 原理']
        assert calls[0][2] == (HIT,)
        assert all('本地参考' not in m['content'] for m in calls[1][1])
        assert [e.kind for e in events[:4]] == ['start','knowledge','delta','delta']
        done = [e for e in events if e.kind == 'done']
        assert done[0].sources == (HIT,) and done[0].retrieval_ms is not None
    finally:
        copilot.close()


def test_slow_retrieval_falls_back_and_late_result_cannot_replace_new_question():
    entered, release, cancelled = threading.Event(), threading.Event(), []
    requests, events = [], []
    class Knowledge:
        notice = 'ready'
        def retrieve(self, query, *, cancel):
            if query == '旧问题':
                entered.set()
                release.wait(3)
                cancelled.append(cancel.is_set())
            return [HIT]
    class Provider:
        async def stream_answer(self, question, context, *, knowledge=()):
            requests.append(question)
            yield '回答'
    copilot = Copilot(Provider(), events.append, knowledge_service=Knowledge())
    try:
        copilot.submit_question('旧问题')
        assert entered.wait(2)
        copilot.submit_question('新问题')
        assert copilot.wait_idle(3)
        release.set()
        assert requests == ['新问题']
        assert not any(e.kind == 'knowledge' and e.question_id == 1 for e in events)
    finally:
        release.set()
        copilot.close()


def test_timeout_or_database_error_keeps_plain_answer():
    class Provider:
        async def stream_answer(self, question, context):
            yield '普通回答'
    class Knowledge:
        notice = ''
        def retrieve(self, query, *, cancel):
            if query == '超时':
                cancel.wait(2)
            else:
                raise RuntimeError('private database path')
            return [HIT]
    events = []
    copilot = Copilot(Provider(), events.append, knowledge_service=Knowledge())
    try:
        for question in ('超时', '错误'):
            copilot.submit_question(question)
            assert copilot.wait_idle(3)
        assert len([e for e in events if e.kind == 'done']) == 2
        slow = next(e for e in events if e.kind == 'done' and e.question == '超时')
        assert slow.elapsed_ms - slow.first_token_ms >= 900
        assert all(not e.sources for e in events if e.kind == 'knowledge')
        assert 'private database' not in str(events)
    finally:
        copilot.close()


@pytest.mark.parametrize('mode,enabled', [('demo', True), ('off', True), ('live', False), ('live', True)])
def test_session_initializes_knowledge_only_when_explicitly_enabled_online(mode, enabled, tmp_path):
    events = []
    config = SessionConfig(Path('capture'), Path('model'), Path('cuda'), microphone=True, device='cpu',
                           answer_mode=mode, api_key='fake-key', knowledge_enabled=enabled,
                           knowledge_directory=tmp_path)
    class Provider:
        async def stream_answer(self, *args, **kwargs):
            yield '测试回答'
    with patch('echomind.knowledge.KnowledgeService') as factory, \
         patch('echomind.desktop_session.OpenAICompatibleProvider', return_value=Provider()), \
         patch('echomind.desktop_session.run_microphone'), \
         patch.object(Path, 'is_file', return_value=True):
        session = DesktopSession(config, lambda *event: events.append(event))
        session.start(None if mode == 'off' else '明确问题')
        assert session.wait(3)
        if mode == 'live' and enabled:
            factory.assert_called_once_with(tmp_path)
            factory.return_value.prepare_async.assert_called_once()
        else:
            factory.assert_not_called()
    assert not any(kind == 'error' for kind, payload in events)
