"""Minimal Tencent Meeting live caption terminal."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


PCM_CHUNK_BYTES = 3200  # 100 ms of 16 kHz mono signed 16-bit audio.
OUTPUT_LOCK = threading.Lock()


def meeting_processes() -> list[dict[str, object]]:
    """Find Tencent Meeting process candidates on Windows."""
    command = (
        "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'WeMeet|Tencent.*Meet|Meet.*Tencent|腾讯会议' "
        "-or $_.ExecutablePath -match 'WeMeet|Tencent.*Meet|Meet.*Tencent|腾讯会议' } | "
        "Select-Object ProcessId, ParentProcessId, Name, ExecutablePath | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "无法列出 Windows 进程")
    if not result.stdout.strip():
        return []
    data = json.loads(result.stdout.lstrip("\ufeff"))
    if isinstance(data, dict):
        data = [data]
    return sorted(data, key=lambda item: int(item["ProcessId"]))


def choose_pid(processes: list[dict[str, object]]) -> int:
    if not processes:
        raise RuntimeError("未找到腾讯会议进程；请先启动会议，或用 --pid 手动指定进程。")
    print("腾讯会议候选进程：")
    for index, process in enumerate(processes, start=1):
        print(
            f"  {index}. PID {process['ProcessId']}  {process['Name']}  "
            f"{process.get('ExecutablePath') or ''}"
        )
    while True:
        answer = input("选择序号（也可直接输入 PID）：").strip()
        if answer.isdecimal():
            number = int(answer)
            if 1 <= number <= len(processes):
                return int(processes[number - 1]["ProcessId"])
            if any(int(item["ProcessId"]) == number for item in processes):
                return number
        print("请输入列表中的序号或 PID。")


def show_events(events: object, copilot=None) -> None:
    for event in events:
        label = "实时" if event.kind == "partial" else "确定"
        receive_to_display_ms = max(0, int(time.monotonic() * 1000) - event.timestamp_ms)
        with OUTPUT_LOCK:
            print(
                f"[{label}] {event.text}  ASR {event.latency_ms:.0f} ms / 接收后显示 {receive_to_display_ms} ms",
                flush=True,
            )
        if copilot is not None:
            copilot.submit_transcript(event.text, event.timestamp_ms, final=event.kind == "final", suspect=event.suspect)


def show_answer(event) -> None:
    with OUTPUT_LOCK:
        if event.kind == "start":
            print(f"\n[问题 #{event.question_id}] {event.question}\n[回答]", flush=True)
        elif event.kind == "delta":
            print(event.text, end="", flush=True)
        elif event.kind == "done":
            first = f"{event.first_token_ms:.0f} ms" if event.first_token_ms is not None else "无内容"
            decision = f" / 语义判断 {event.decision_ms:.0f} ms" if event.decision_ms is not None else ""
            print(f"\n[完成 #{event.question_id}] 模型首字 {first} / 总计 {event.elapsed_ms:.0f} ms{decision}", flush=True)
        elif event.kind == "checking":
            print(f"\n[语义判断] {event.question}", flush=True)
        elif event.kind == "reviewing":
            print(f"\n[问题复核] {event.question}", flush=True)
        elif event.kind == "needs_confirmation":
            print(f"[需确认] 原文：{event.question}\n建议：{event.text}\n{event.reason}\n请在窗口版中编辑核对后提交。", flush=True)
        elif event.kind == "repair_error":
            print(f"[复核失败] {event.text}", file=sys.stderr, flush=True)
        elif event.kind == "skipped":
            print(f"[忽略] {event.text}", flush=True)
        elif event.kind == "gate_error":
            print(f"[判断失败] {event.text}", file=sys.stderr, flush=True)
        elif event.kind == "cancelled":
            print(f"\n[已取消 #{event.question_id}]", flush=True)
        elif event.kind == "error":
            print(f"\n[回答失败 #{event.question_id}] {event.text}", file=sys.stderr, flush=True)


def run(pid: int, capture_exe: Path, model: str, device: str, compute_type: str, copilot=None,
        *, stop_event=None, on_transcript=None, on_status=None) -> int:
    if not capture_exe.is_file():
        raise RuntimeError(f"采集程序不存在：{capture_exe}。请先构建 native/audio-capture。")
    from .asr import Transcriber

    transcriber = Transcriber(model_size=model, device=device, compute_type=compute_type)
    stopping = stop_event if stop_event is not None else threading.Event()
    gui_mode = on_transcript is not None or on_status is not None

    def status(text):
        if on_status is not None:
            on_status(text)
        elif not gui_mode:
            print(text, flush=True)

    def publish(events):
        if not gui_mode:
            show_events(events, copilot)
        else:
            for event in events:
                if on_transcript is not None:
                    on_transcript(event)
                if copilot is not None:
                    copilot.submit_transcript(event.text, event.timestamp_ms, final=event.kind == "final", suspect=event.suspect)

    status("正在预热识别模型…")
    transcriber.warm_up()
    if stopping.is_set():
        return 0
    process = subprocess.Popen(
        [str(capture_exe), "--pid", str(pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if gui_mode else None,
        bufsize=PCM_CHUNK_BYTES * 4,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if gui_mode and os.name == "nt" else {}),
    )
    status(f"正在采集会议声音（PID {pid}）" if gui_mode else f"采集 PID {pid}；按 Ctrl+C 退出。")
    chunks: queue.Queue[tuple[bytes, int] | None] = queue.Queue(maxsize=100)

    def drain_stderr():
        # Do not expose child diagnostic text, which is not a trusted UI message.
        if process.stderr is not None:
            while process.stderr.read(4096):
                pass

    stderr_reader = threading.Thread(target=drain_stderr, daemon=True) if gui_mode else None
    if stderr_reader is not None:
        stderr_reader.start()

    def read_audio() -> None:
        assert process.stdout is not None
        while not stopping.is_set():
            pcm = process.stdout.read(PCM_CHUNK_BYTES)
            if not pcm:
                break
            item = (pcm, int(time.monotonic() * 1000))
            while not stopping.is_set():
                try:
                    chunks.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
        while not stopping.is_set():
            try:
                chunks.put(None, timeout=0.1)
                break
            except queue.Full:
                continue

    reader = threading.Thread(target=read_audio, daemon=True)
    reader.start()
    try:
        while not stopping.is_set():
            try:
                item = chunks.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            pcm, timestamp_ms = item
            publish(transcriber.push_pcm(pcm, timestamp_ms=timestamp_ms))
        if stopping.is_set():
            return 0
        publish(transcriber.flush())
        while not stopping.is_set():
            try:
                returncode = process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            return 0
        if returncode:
            raise RuntimeError(f"音频采集程序已退出（退出码 {returncode}）。")
        return 0
    finally:
        stopping.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=1)
        if process.stdout is not None:
            process.stdout.close()
        if stderr_reader is not None:
            stderr_reader.join(timeout=1)
        if gui_mode and process.stderr is not None:
            process.stderr.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EchoMind 腾讯会议实时中文字幕")
    parser.add_argument("--pid", type=int, help="腾讯会议的目标进程 PID；不指定则交互选择")
    parser.add_argument("--capture-exe", type=Path, help="audio-capture.exe 路径")
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper 模型")
    parser.add_argument("--device", default="auto", help="ASR 设备，例如 cuda 或 cpu")
    parser.add_argument("--compute-type", default="default", help="ASR 计算精度")
    parser.add_argument("--answers", choices=("off", "demo", "live"), default="off",
                        help="off 仅字幕；demo 静态演练；live 使用配置的回答 API")
    parser.add_argument("--text-question", help="直接测试一个中文问题，不启动音频和 ASR")
    parser.add_argument("--question-gate", choices=("rules", "jev"), default="rules",
                        help="rules 本地过滤；jev 使用 TypeSafe 语义确认，需要单独密钥")
    parser.add_argument("--question-threshold", type=float, default=0.85,
                        help="Jev 提问概率触发门限，默认 0.85")
    args = parser.parse_args(argv)
    copilot = None
    answer_failed = threading.Event()
    answered = threading.Event()
    skipped = threading.Event()

    def on_answer(event) -> None:
        show_answer(event)
        if event.kind in ("error", "gate_error", "repair_error"):
            answer_failed.set()
        elif event.kind == "done":
            answered.set()
        elif event.kind in ("skipped", "needs_confirmation"):
            skipped.set()

    try:
        if args.pid is not None and args.pid <= 0:
            parser.error("--pid 必须为正整数")
        if args.text_question is not None and args.answers == "off":
            parser.error("--text-question 需要 --answers demo 或 live")
        if args.text_question is None and args.capture_exe is None:
            parser.error("音频模式需要 --capture-exe")
        if args.question_gate == "jev" and args.answers != "live":
            parser.error("--question-gate jev 需要 --answers live（会调用在线判断接口）")
        if args.answers != "off":
            from .copilot import Copilot
            from .llm import DemoProvider, OpenAICompatibleProvider

            provider = DemoProvider() if args.answers == "demo" else OpenAICompatibleProvider.from_env()
            gate = None
            if args.question_gate == "jev":
                from .intent import JevQuestionGate
                gate = JevQuestionGate.from_env()
                print("Jev 语义补判已启用：规则未命中的确定字幕和最近最多 6 段字幕将发送至 TypeSafe。", flush=True)
            if args.answers == "live":
                print("在线回答已启用：问题和最近回答上下文将发送至配置的 API。", flush=True)
            else:
                print("演练模式：固定示例文本，不调用在线模型。", flush=True)
            copilot = Copilot(provider, on_answer, question_gate=gate, question_threshold=args.question_threshold)
        if args.text_question is not None:
            copilot.submit_transcript(args.text_question, int(time.monotonic() * 1000))
            if not copilot.wait_idle(45):
                raise RuntimeError("回答测试超时，已停止生成。")
            if not answered.is_set() and not answer_failed.is_set() and not skipped.is_set():
                raise RuntimeError("未识别为完整问题；请使用“为什么／如何／请介绍”等明确问法。")
            return 1 if answer_failed.is_set() else 0
        pid = args.pid if args.pid is not None else choose_pid(meeting_processes())
        return run(pid, args.capture_exe, args.model, args.device, args.compute_type, copilot)
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    finally:
        if copilot is not None:
            copilot.close()


if __name__ == "__main__":
    raise SystemExit(main())
