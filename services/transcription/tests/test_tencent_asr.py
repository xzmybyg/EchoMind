"""Protocol tests never connect to Tencent or use real credentials/audio."""

import base64
from collections import deque
from dataclasses import replace
import hashlib
import hmac
import json
import threading
import time
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

import pytest

from echomind.tencent_asr import TencentConfig, TencentTranscriber, TencentASRError, signed_url


CONFIG = TencentConfig("123456", "test-id", "test-secret", consent=True)


class Socket:
    def __init__(self):
        self.messages = deque([json.dumps({"code": 0})])
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(data)
        if isinstance(data, str):
            self.messages.append(json.dumps({"code": 0, "final": 1}))

    def recv(self, timeout=0):
        if not self.messages:
            raise TimeoutError()
        return self.messages.popleft()

    def close(self):
        self.closed = True


@pytest.fixture
def cloud(monkeypatch):
    socket = Socket()
    connect = Mock(return_value=socket)
    monkeypatch.setattr("websockets.sync.client.connect", connect)
    transcriber = TencentTranscriber(CONFIG)
    transcriber.warm_up()
    yield transcriber, socket, connect
    transcriber.close()


def result(text, kind=2, index=0, end=200):
    return json.dumps({"code": 0, "result": {"index": index, "slice_type": kind,
                      "end_time": end, "voice_text_str": text}})


def test_signature_uses_sorted_raw_parameters_then_url_encoding():
    url = signed_url(CONFIG, "React Hooks, 灰度发布", now=100, nonce=42, voice_id="voice")
    parsed = urlsplit(url)
    query = {key: value[0] for key, value in parse_qs(parsed.query).items()}
    signature = query.pop("signature")
    canonical = ("asr.cloud.tencent.com/asr/v2/123456?convert_num_mode=1&engine_model_type=16k_zh_en"
                 "&expired=3700&filter_empty_result=1&hotword_list=React|5,Hooks|5,灰度发布|5&needvad=1"
                 "&nonce=42&secretid=test-id&timestamp=100&vad_silence_time=1000&voice_format=1&voice_id=voice")
    expected = base64.b64encode(hmac.new(b"test-secret", canonical.encode(), hashlib.sha1).digest()).decode()
    assert signature == expected
    assert "test-secret" not in url
    assert "%E7%81%B0" in url and "%7C" in url
    assert query["voice_format"] == "1"


def test_unique_voice_id_for_each_connection():
    assert parse_qs(urlsplit(signed_url(CONFIG)).query)["voice_id"] != parse_qs(urlsplit(signed_url(CONFIG)).query)["voice_id"]


@pytest.mark.parametrize("changes, message", [({"consent": False}, "同意"), ({"app_id": "bad"}, "AppID"),
    ({"secret_key": ""}, "SecretKey"), ({"secret_id": ""}, "SecretId"), ({"engine": "unknown"}, "模型")])
def test_configuration_validated_before_network(changes, message, monkeypatch):
    connect = Mock()
    monkeypatch.setattr("websockets.sync.client.connect", connect)
    with pytest.raises(TencentASRError, match=message):
        TencentTranscriber(replace(CONFIG, **changes)).warm_up()
    connect.assert_not_called()


def test_credentials_not_in_repr():
    assert "test-secret" not in repr(CONFIG) and "test-id" not in repr(CONFIG)


def test_packet_size_stop_and_end_marker(cloud):
    t, socket, connect = cloud
    t.push_pcm(bytes(3200), timestamp_ms=1000)
    assert socket.sent == []
    t.push_pcm(bytes(3200), timestamp_ms=1100)
    assert socket.sent == [bytes(6400)]
    t.push_pcm(bytes(100), timestamp_ms=1104)
    t.stop_event.set()
    t.push_pcm(bytes(6400))
    t.flush()
    assert socket.sent == [bytes(6400), '{"type":"end"}']
    t.close()
    assert socket.closed and not t._pending
    assert connect.call_args.kwargs["logger"].disabled
    assert connect.call_args.kwargs["close_timeout"] == 1


def test_frame_pacing_and_normal_tail_flush(cloud):
    t, socket, _ = cloud
    started = time.monotonic()
    t.push_pcm(bytes(12800), timestamp_ms=400)
    assert time.monotonic() - started >= 0.18
    t.push_pcm(bytes(100))
    t.flush()
    assert socket.sent == [bytes(6400), bytes(6400), bytes(100), '{"type":"end"}']


def test_streaming_simplified_dedup_and_timestamps(cloud):
    t, socket, _ = cloud
    t.push_pcm(bytes(6400), timestamp_ms=1200)
    socket.messages.extend([result("請解釋", 1), result("請解釋", 1), result("請解釋 React Hooks", 2),
                            result("請解釋 React Hooks", 2), result("舊的字幕", 1)])
    events = t.poll()
    assert [(e.kind, e.text) for e in events] == [("partial", "请解释"), ("final", "请解释 React Hooks")]
    assert events[-1].timestamp_ms == 1200


def test_queued_final_while_capture_idle_is_delivered(cloud):
    t, socket, _ = cloud
    socket.messages.append(result("为什么"))
    assert t.poll()[0].kind == "final"


def test_repetitive_output_rejected(cloud):
    t, socket, _ = cloud
    socket.messages.append(result("Hooks, Hooks, Hooks, Hooks"))
    event = t.poll()[0]
    assert event.kind == "rejected" and event.suspect


@pytest.mark.parametrize("code", [4002, 4003, 4004, 4005, 4006, 4008, 5000, 9876])
def test_remote_errors_are_safe_and_actionable(cloud, code):
    t, socket, _ = cloud
    socket.messages.append(json.dumps({"code": code, "message": "test-secret signed-url"}))
    with pytest.raises(TencentASRError) as exc:
        t.poll()
    assert str(code) in str(exc.value)
    assert "test-secret" not in str(exc.value) and "signed-url" not in str(exc.value)


@pytest.mark.parametrize("raw", ["bad-json", "[]", '{"code":"0"}', '{"code":0,"result":{"index":1}}'])
def test_malformed_responses_do_not_leak(cloud, raw):
    t, socket, _ = cloud
    socket.messages.append(raw)
    with pytest.raises(TencentASRError, match="格式异常"):
        t.poll()


def test_handshake_failure_closes_socket(monkeypatch):
    socket = Socket()
    socket.messages = deque(['{"code":4002,"message":"test-secret"}'])
    monkeypatch.setattr("websockets.sync.client.connect", Mock(return_value=socket))
    with pytest.raises(TencentASRError, match="鉴权失败"):
        TencentTranscriber(CONFIG).warm_up()
    assert socket.closed


def test_connection_failure_safe_and_cancel_skips_connect(monkeypatch):
    connect = Mock(side_effect=RuntimeError("test-secret"))
    monkeypatch.setattr("websockets.sync.client.connect", connect)
    with pytest.raises(TencentASRError, match="无法连接") as exc:
        TencentTranscriber(CONFIG).warm_up()
    assert "test-secret" not in str(exc.value)
    connect.reset_mock()
    stop = threading.Event()
    stop.set()
    TencentTranscriber(CONFIG, stop_event=stop).warm_up()
    connect.assert_not_called()


def test_disconnect_and_unexpected_final_fail_clearly(cloud):
    t, socket, _ = cloud
    socket.messages.append('{"code":0,"final":1}')
    with pytest.raises(TencentASRError, match="提前结束"):
        t.poll()
    with pytest.raises(TencentASRError, match="已结束"):
        t.push_pcm(bytes(6400))


def test_no_final_ack_is_bounded(cloud, monkeypatch):
    t, socket, _ = cloud
    socket.send = lambda _: None
    ticks = iter([0, 1, 2, 4])
    monkeypatch.setattr("echomind.tencent_asr.time.monotonic", lambda: next(ticks))
    with pytest.raises(TencentASRError, match="超时"):
        t.flush()


@pytest.mark.parametrize("words", ["字" * 11, "x" * 31, "React|100", "a&b", ",".join(f"word{i}" for i in range(129))])
def test_invalid_hotwords_rejected_without_request(words):
    with pytest.raises(TencentASRError, match="热词"):
        signed_url(CONFIG, words)


def test_real_websocket_stream_against_local_protocol_server(monkeypatch):
    """Exercise the installed library with loopback only, including delayed final."""
    from websockets.sync.server import serve
    from websockets.sync.client import connect as real_connect
    received = []

    def handle(connection):
        connection.send('{"code":0}')
        for data in connection:
            received.append(data)
            if isinstance(data, bytes):
                connection.send(result("React", kind=1))
            else:
                connection.send(result("讲解一下 React Hooks", kind=2))
                connection.send('{"code":0,"final":1}')
                break

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        monkeypatch.setattr("websockets.sync.client.connect", lambda _url, **kwargs: real_connect(address, proxy=None, **kwargs))
        t = TencentTranscriber(CONFIG)
        try:
            t.warm_up()
            events = t.push_pcm(bytes(6400), timestamp_ms=200)
            events.extend(t.poll(timeout=1))
            events.extend(t.flush())
            assert [event.kind for event in events] == ["partial", "final"]
            assert events[-1].text == "讲解一下 React Hooks"
            assert received == [bytes(6400), '{"type":"end"}']
        finally:
            t.close()
            server.shutdown()
            thread.join(2)


def test_connection_drop_does_not_expose_remote_exception(cloud):
    t, socket, _ = cloud
    socket.recv = Mock(side_effect=RuntimeError("test-secret"))
    with pytest.raises(TencentASRError, match="连接中断") as exc:
        t.poll()
    assert "test-secret" not in str(exc.value)
