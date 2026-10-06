"""Threaded desktop session; credentials stay in the caller's memory."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from collections.abc import Callable

import httpx

from . import cli
from .microphone import MicrophoneError, run_microphone
from .copilot import Copilot
from .intent import JevQuestionGate
from .llm import DemoProvider, OpenAICompatibleProvider
from .repair import OpenAICompatibleQuestionReviser


@dataclass(frozen=True)
class SessionConfig:
    capture_exe: Path
    model_path: Path
    cuda_dir: Path
    pid: int = 0
    device: str = "cuda"
    answer_mode: str = "off"
    question_gate: str = "rules"
    api_base_url: str = "https://api.deepseek.com"
    api_model: str = "deepseek-flash"
    api_key: str = field(default="", repr=False)
    jev_key: str = field(default="", repr=False)
    microphone: bool = False


class DesktopSession:
    def __init__(self, config: SessionConfig, on_event: Callable[[str, object], None]):
        self.config = config
        self._on_event = on_event
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._copilot: Copilot | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _validate(self, question: str | None):
        config = self.config
        if config.answer_mode not in ("off", "demo", "live"):
            raise ValueError("回答模式无效")
        if config.question_gate not in ("rules", "jev"):
            raise ValueError("问题判断模式无效")
        if config.device not in ("cuda", "cpu"):
            raise ValueError("识别设备无效")
        if config.answer_mode == "live":
            if not config.api_key.strip():
                raise ValueError("请填写回答模型 API Key")
            if not config.api_model.strip():
                raise ValueError("请填写回答模型名称")
            try:
                url = httpx.URL(config.api_base_url)
            except httpx.InvalidURL:
                raise ValueError("回答接口地址格式无效") from None
            if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
                raise ValueError("回答接口必须是无凭据、无查询参数的 HTTP(S) 地址")
        if config.question_gate == "jev":
            if config.answer_mode != "live":
                raise ValueError("Jev 判断需要在线回答模式")
            if not config.jev_key.strip():
                raise ValueError("请填写 Jev API Key")
        if question is not None:
            if not question.strip():
                raise ValueError("请填写测试问题")
            if config.answer_mode == "off":
                raise ValueError("测试回答需要演练或在线回答模式")
        else:
            if not config.microphone and config.pid <= 0:
                raise ValueError("请先选择腾讯会议进程")
            if not config.microphone and not config.capture_exe.is_file():
                raise ValueError("未找到会议采集程序，请检查程序文件夹")
            if not (config.model_path / "model.bin").is_file():
                raise ValueError("未找到语音识别模型，请检查模型文件夹")
            if config.device == "cuda" and not (config.cuda_dir / "cublas64_12.dll").is_file():
                raise ValueError("未找到 GPU 运行库，请检查 cuda 文件夹或改用 CPU")

    def start(self, question: str | None = None) -> None:
        with self._lock:
            if self._running:
                raise ValueError("当前会话尚未停止")
            self._validate(question)
            self._stop.clear()
            self._running = True
            self._thread = threading.Thread(target=self._work, args=(question,), name="echomind-session", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()

    def submit_question(self, text: str) -> None:
        with self._lock:
            if self.config.answer_mode == "off":
                raise ValueError("仅字幕模式不生成回答，请选择演练或在线回答")
            if not self._running or self._stop.is_set():
                raise RuntimeError("采集会话未运行或正在停止，请重新开始采集")
            if self._copilot is None:
                raise RuntimeError("回答服务尚未就绪，请稍后重试")
            self._copilot.submit_question(text)

    def wait(self, timeout: float) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_running

    def _work(self, question: str | None):
        copilot = None
        dll_handle = None
        old_path = None
        config = self.config
        def on_answer(event):
            self._on_event("answer", event)

        try:
            if self._stop.is_set():
                return
            if config.answer_mode != "off":
                provider = DemoProvider() if config.answer_mode == "demo" else OpenAICompatibleProvider(
                    config.api_base_url.strip(), config.api_model.strip(), config.api_key.strip(),
                    thinking="disabled" if "deepseek" in (httpx.URL(config.api_base_url).host or "").lower() else None,
                )
                gate = JevQuestionGate(config.jev_key.strip()) if config.question_gate == "jev" else None
                reviser = OpenAICompatibleQuestionReviser(
                    config.api_base_url.strip(), config.api_model.strip(), config.api_key.strip(),
                ) if config.answer_mode == "live" else None
                copilot = Copilot(provider, on_answer, question_gate=gate, question_reviser=reviser)
                if question is None:
                    with self._lock:
                        if not self._stop.is_set():
                            self._copilot = copilot
            if question is not None:
                self._on_event("status", "正在测试回答…")
                copilot.submit_question(question)
                deadline = time.monotonic() + 45
                while not self._stop.is_set():
                    if copilot.wait_idle(0.1):
                        break
                    if time.monotonic() >= deadline:
                        self._on_event("error", "回答测试超时，已停止生成")
                        break
            else:
                if config.device == "cuda":
                    old_path = os.environ.get("PATH", "")
                    os.environ["PATH"] = str(config.cuda_dir) + os.pathsep + old_path
                    if os.name == "nt":
                        dll_handle = os.add_dll_directory(str(config.cuda_dir))
                callbacks = dict(stop_event=self._stop,
                                 on_transcript=lambda event: self._on_event("transcript", event),
                                 on_status=lambda message: self._on_event("status", message))
                compute = "int8_float16" if config.device == "cuda" else "int8"
                if config.microphone:
                    run_microphone(str(config.model_path), config.device, compute, copilot, **callbacks)
                else:
                    cli.run(config.pid, config.capture_exe, str(config.model_path), config.device, compute, copilot, **callbacks)
        except MicrophoneError as error:
            self._on_event("error", str(error))
        except Exception as error:
            # Never show exception strings, remote bodies or credentials in the UI.
            self._on_event("error", f"{type(error).__name__}：会话未能运行，请检查模型、采集程序和配置")
        finally:
            with self._lock:
                self._copilot = None
            try:
                if copilot is not None:
                    copilot.close()
            except Exception as error:
                self._on_event("error", f"{type(error).__name__}：后台连接关闭失败")
            finally:
                if dll_handle is not None:
                    dll_handle.close()
                if old_path is not None:
                    os.environ["PATH"] = old_path
                with self._lock:
                    self._running = False
                self._on_event("finished", None)
