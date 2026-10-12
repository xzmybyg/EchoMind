"""VAD-gated Qwen captions using an isolated local GPU worker."""
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import monotonic, perf_counter

from .asr import Transcriber, _decode_anomaly
from .text import normalize_transcript


class QwenASRError(RuntimeError):
    pass


def runtime_python(model_path):
    from .components import cache_root
    app = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[3]
    roots = [app, *Path(model_path).resolve().parents[:4], cache_root()]
    candidates = [root / name / "qwen17-eval/Scripts/python.exe"
                  for root in roots for name in (".runtime", "runtime")]
    for path in candidates:
        if path.is_file():
            return path
    raise QwenASRError("未找到 Qwen 独立运行环境，请保留项目中的 .runtime/qwen17-eval，或将它部署到程序旁 runtime/qwen17-eval。")


class QwenTranscriber(Transcriber):
    def __init__(self, *, stop_event=None, **options):
        # No word timestamps: never mix previous utterances into the decoder.
        options.update(context_window_ms=0, retry_low_confidence=False, hotwords=None, wait_for_continuation=True)
        options["partial_interval_ms"] = options.get("max_utterance_ms", 12000) + 1000
        if options.get("vad") is None:
            from .vad import SpeechVAD
            options["vad"] = SpeechVAD()
        super().__init__(**options)
        self._stop = stop_event or threading.Event()
        self._process = None
        self._reader = None
        self._responses = queue.Queue()

    def _receive(self, timeout):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self._stop.is_set():
                self.close()
                return None
            try:
                response = self._responses.get(timeout=0.1)
            except queue.Empty:
                continue
            if not isinstance(response, dict) or response.get("error"):
                self.close()
                raise QwenASRError("Qwen 识别进程异常，请检查独立运行环境、模型或 GPU 显存后重试。")
            return response
        self.close()
        raise QwenASRError("Qwen 加载或识别超时，请检查 GPU 负载后重试。")

    def _load(self):
        if self._process is not None or self._stop.is_set():
            return
        python = runtime_python(self.model_size)
        env = os.environ.copy()
        # Do not leak the frozen application's DLL/search paths into Python.
        for key in ("PYTHONHOME", "PYTHONPATH", "_MEIPASS2"):
            env.pop(key, None)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
        self._process = subprocess.Popen(
            [str(python), "-I", "-X", "utf8", str(Path(__file__).with_name("qwen_worker.py")), str(Path(self.model_size).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        stdout = self._process.stdout

        def read():
            try:
                for line in stdout:
                    self._responses.put(json.loads(line))
            except (ValueError, OSError):
                pass
            finally:
                self._responses.put(None)

        self._reader = threading.Thread(target=read, name="qwen-output", daemon=True)
        self._reader.start()
        response = self._receive(120)
        if response is not None and not response.get("ready"):
            self.close()
            raise QwenASRError("Qwen 模型初始化失败。")

    def _transcribe(self, pcm, *, final=False, warmup=False, **kwargs):
        started = perf_counter()
        self._last_suspect = False
        self._last_original = self._last_rejection = ""
        self._last_word_end_ms = 0.0
        if self._stop.is_set() or (not warmup and not any(pcm)):
            return "", 0.0
        self._load()
        if self._stop.is_set():
            return "", 0.0
        try:
            self._process.stdin.write(json.dumps({"pcm": base64.b64encode(pcm).decode("ascii")}) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.close()
            raise QwenASRError("Qwen 识别进程已退出，请重新开始采集。") from None
        response = self._receive(30)
        text = normalize_transcript(response.get("text", "")) if response else ""
        self._last_rejection = _decode_anomaly(text)
        if response and response.get("truncated"):
            self._last_rejection = "识别输出达到长度上限，请缩短发言后重试"
        self._last_suspect = bool(self._last_rejection)
        return text, (perf_counter() - started) * 1000

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._reader:
            self._reader.join(timeout=1)
        process.stdin.close()
        process.stdout.close()
