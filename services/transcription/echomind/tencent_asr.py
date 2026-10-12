"""Tencent streaming ASR. Audio is uploaded only after an explicit cloud opt-in."""

import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import logging
import re
import secrets
import threading
import time
from urllib.parse import urlencode
import uuid

from .asr import TranscriptEvent, _decode_anomaly
from .text import normalize_transcript


class TencentASRError(RuntimeError):
    """Only safe, locally authored messages; never remote bodies or signed URLs."""


@dataclass(frozen=True)
class TencentConfig:
    app_id: str = ""
    secret_id: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    engine: str = "16k_zh_en"
    consent: bool = False

    def validate(self):
        if not self.consent:
            raise TencentASRError("请先同意将采集音频上传腾讯云识别（可能产生费用）。")
        if not re.fullmatch(r"[0-9]+", self.app_id):
            raise TencentASRError("请填写腾讯云数字 AppID。")
        if not self.secret_id.strip() or not self.secret_key.strip():
            raise TencentASRError("请填写腾讯云 SecretId 和 SecretKey。")
        if self.engine not in ("16k_zh_en", "16k_zh"):
            raise TencentASRError("请选择支持的腾讯云识别模型。")


def signed_url(config, hotwords="", *, now=None, nonce=None, voice_id=None):
    config.validate()
    now = int(time.time()) if now is None else now
    params = dict(engine_model_type=config.engine, expired=now + 3600,
                  needvad=1, nonce=secrets.randbelow(10**9) if nonce is None else nonce,
                  secretid=config.secret_id, timestamp=now, voice_format=1,
                  voice_id=voice_id or uuid.uuid4().hex, vad_silence_time=1000,
                  filter_empty_result=1, convert_num_mode=1)
    # Tencent disallows spaces within a hotword; use modest weights, not forced replacement.
    words = list(dict.fromkeys(word for word in re.split(r"[,，\s]+", hotwords) if word))
    if words:
        if len(words) > 128 or any(len(word) > 30 or len(re.findall(r"[\u4e00-\u9fff]", word)) > 10 or re.search(r"[|&=]", word) for word in words):
            raise TencentASRError("腾讯云热词最多 128 个，每个不超过 30 字符或 10 个汉字，不能含 |、&、=。")
        params["hotword_list"] = ",".join(f"{word}|5" for word in words)
    path = f"asr.cloud.tencent.com/asr/v2/{config.app_id}"
    ordered = sorted(params.items())
    original = path + "?" + "&".join(f"{key}={value}" for key, value in ordered)
    signature = base64.b64encode(hmac.new(config.secret_key.encode(), original.encode(), hashlib.sha1).digest()).decode()
    return "wss://" + path + "?" + urlencode([*ordered, ("signature", signature)])


def _message(raw):
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get("code")) is not int:
            raise ValueError()
    except (ValueError, TypeError):
        raise TencentASRError("腾讯云识别返回格式异常，请停止后重试。") from None
    code = data["code"]
    if code:
        reason = {4001: "参数不合法，请检查模型和热词", 4002: "鉴权失败，请检查凭证和系统时间",
                  4003: "尚未开通语音识别服务", 4004: "资源包耗尽，请检查计费配置",
                  4005: "账户欠费", 4006: "并发连接超限", 4007: "音频格式错误",
                  4008: "音频发送超时", 4000: "音频发送过快", 5000: "服务暂时不可用"}.get(code, "服务拒绝请求，请检查控制台")
        raise TencentASRError(f"腾讯云识别错误 {code}：{reason}。")
    return data


class TencentTranscriber:
    """Same push/poll/flush contract as local ASR; network stays on the session worker."""

    def __init__(self, config, *, hotwords="", stop_event=None):
        self.config = config
        self.hotwords = hotwords or ""
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self._socket = None
        self._pending = bytearray()
        self._origin_ms = None
        self._last_send = 0.0
        self._final_index = -1
        self._last_partial = None
        self._ended = False
        self._finishing = False

    def warm_up(self):
        from websockets.sync.client import connect
        url = signed_url(self.config, self.hotwords)
        if self.stop_event.is_set():
            return
        # The library's debug log includes the signed handshake URL and audio frames.
        logger = logging.Logger("echomind.cloud_asr.private")
        logger.disabled = True
        try:
            self._socket = connect(url, open_timeout=5, close_timeout=1,
                                   compression=None, max_size=1_048_576, logger=logger)
            _message(self._socket.recv(timeout=5))
        except TencentASRError:
            self.close()
            raise
        except Exception:
            self.close()
            raise TencentASRError("无法连接腾讯云识别，请检查网络、代理及系统时间后重试。") from None

    def _send(self, data):
        try:
            self._socket.send(data)
        except Exception:
            raise TencentASRError("腾讯云识别连接已断开，请停止后重新开始。") from None

    def push_pcm(self, pcm, *, timestamp_ms=None):
        if self.stop_event.is_set():
            return []
        if self._ended or self._finishing:
            raise TencentASRError("腾讯云识别已结束，请重新开始采集。")
        if len(pcm) % 2:
            raise TencentASRError("腾讯云识别需要 16 位单声道 PCM 音频。")
        if self._origin_ms is None:
            end = time.monotonic() * 1000 if timestamp_ms is None else timestamp_ms
            self._origin_ms = end - len(pcm) / 32
        self._pending.extend(pcm)
        events = self.poll()
        while len(self._pending) >= 6400:  # 200 ms at 16 kHz / int16.
            if self.stop_event.wait(max(0, self._last_send + 0.2 - time.monotonic())):
                break
            self._send(bytes(self._pending[:6400]))
            del self._pending[:6400]
            self._last_send = time.monotonic()
            events.extend(self.poll())
        return events

    def poll(self, timeout=0):
        events = []
        while not self._ended and self._socket is not None:
            try:
                raw = self._socket.recv(timeout=timeout)
            except TimeoutError:
                break
            except Exception:
                raise TencentASRError("腾讯云识别连接中断，请检查网络后重新开始。") from None
            timeout = 0
            data = _message(raw)
            result = data.get("result")
            if result:
                try:
                    index, kind = int(result["index"]), int(result["slice_type"])
                    end_ms = int(result["end_time"])
                    text = normalize_transcript(result["voice_text_str"])
                    if kind not in (0, 1, 2) or index < 0 or end_ms < 0:
                        raise ValueError()
                except (KeyError, ValueError, TypeError, AttributeError):
                    raise TencentASRError("腾讯云字幕格式异常，请停止后重试。") from None
                if index > self._final_index:
                    if kind == 2:
                        self._final_index = index
                    key = (index, text)
                    if text and (kind == 2 or key != self._last_partial):
                        anomaly = _decode_anomaly(text)
                        timestamp = int((self._origin_ms or 0) + end_ms)
                        events.append(TranscriptEvent(
                            "rejected" if anomaly else "final" if kind == 2 else "partial",
                            text, timestamp, max(0, time.monotonic() * 1000 - timestamp),
                            suspect=bool(anomaly), rejection_reason=anomaly))
                        self._last_partial = key
            if data.get("final") == 1:
                self._ended = True
                if not self._finishing:
                    raise TencentASRError("腾讯云识别提前结束，请重新开始采集。")
        return events

    def flush(self):
        if self._socket is None or self._ended:
            return []
        self._finishing = True
        if self._pending:
            # A final short frame is valid; do not send buffered audio after Stop.
            if not self.stop_event.is_set():
                time.sleep(max(0, self._last_send + 0.2 - time.monotonic()))
                self._send(bytes(self._pending))
            self._pending.clear()
        self._send('{"type":"end"}')
        deadline = time.monotonic() + 3
        events = []
        while not self._ended and time.monotonic() < deadline:
            events.extend(self.poll(timeout=min(0.1, max(0, deadline - time.monotonic()))))
        if not self._ended:
            raise TencentASRError("腾讯云结束识别超时，最后一句可能不完整，请重试。")
        return events

    def close(self):
        connection, self._socket = self._socket, None
        self._pending.clear()
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
