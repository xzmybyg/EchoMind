"""Minimal Tencent Meeting live caption terminal."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


PCM_CHUNK_BYTES = 3200  # 100 ms of 16 kHz mono signed 16-bit audio.


def meeting_processes() -> list[dict[str, object]]:
    """Find Tencent Meeting process candidates on Windows."""
    command = (
        "[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match 'WeMeet|Tencent.*Meet|Meet.*Tencent|腾讯会议' "
        "-or $_.ExecutablePath -match 'WeMeet|Tencent.*Meet|Meet.*Tencent|腾讯会议' } | "
        "Select-Object ProcessId, Name, ExecutablePath | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
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


def show_events(events: object) -> None:
    for event in events:
        label = "实时" if event.kind == "partial" else "确定"
        receive_to_display_ms = max(0, int(time.monotonic() * 1000) - event.timestamp_ms)
        print(
            f"[{label}] {event.text}  ASR {event.latency_ms:.0f} ms / 接收后显示 {receive_to_display_ms} ms",
            flush=True,
        )


def run(pid: int, capture_exe: Path, model: str, device: str, compute_type: str) -> int:
    if not capture_exe.is_file():
        raise RuntimeError(f"采集程序不存在：{capture_exe}。请先构建 native/audio-capture。")
    from .asr import Transcriber

    transcriber = Transcriber(model_size=model, device=device, compute_type=compute_type)
    print("正在预热识别模型…", flush=True)
    transcriber.warm_up()
    process = subprocess.Popen(
        [str(capture_exe), "--pid", str(pid)],
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=PCM_CHUNK_BYTES * 4,
    )
    print(f"采集 PID {pid}；按 Ctrl+C 退出。", flush=True)
    chunks: queue.Queue[tuple[bytes, int] | None] = queue.Queue(maxsize=100)
    stopping = threading.Event()

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
        while True:
            item = chunks.get()
            if item is None:
                break
            pcm, timestamp_ms = item
            show_events(transcriber.push_pcm(pcm, timestamp_ms=timestamp_ms))
        show_events(transcriber.flush())
        returncode = process.wait()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EchoMind 腾讯会议实时中文字幕")
    parser.add_argument("--pid", type=int, help="腾讯会议的目标进程 PID；不指定则交互选择")
    parser.add_argument("--capture-exe", type=Path, required=True, help="audio-capture.exe 路径")
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper 模型")
    parser.add_argument("--device", default="auto", help="ASR 设备，例如 cuda 或 cpu")
    parser.add_argument("--compute-type", default="default", help="ASR 计算精度")
    args = parser.parse_args(argv)
    try:
        if args.pid is not None and args.pid <= 0:
            parser.error("--pid 必须为正整数")
        pid = args.pid if args.pid is not None else choose_pid(meeting_processes())
        return run(pid, args.capture_exe, args.model, args.device, args.compute_type)
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
