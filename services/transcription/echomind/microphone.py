"""Default microphone input; capture callback never runs Whisper or network IO."""

import queue
import time


class MicrophoneError(RuntimeError):
    pass


def input_quality(pcm):
    """Describe the captured signal before any ASR gain is applied."""
    import numpy as np
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    level = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    if samples.size and float(np.mean(np.abs(samples) >= 0.998)) >= 0.01:
        return level, "声音削波，请降低麦克风增益"
    if level < 0.0001:
        return level, "未检测到输入声音"
    if level < 0.003:
        return level, "输入较低；说话时请靠近麦克风"
    return level, ""


def run_microphone(model, device, compute_type, copilot=None, *, stop_event, on_transcript, on_status, asr_options=None):
    import av
    import numpy as np
    import sounddevice as sd
    from .asr_backends import create_transcriber

    transcriber = create_transcriber(model_size=model, device=device, compute_type=compute_type, **(asr_options or {}))
    cloud = (asr_options or {}).get("engine") == "tencent"
    on_status("正在连接腾讯云识别…" if cloud else "正在预热识别模型…")
    transcriber.warm_up()
    if stop_event.is_set():
        return
    try:
        info = sd.query_devices(kind="input")
        rate = int(info["default_samplerate"])
        if rate <= 0 or info["max_input_channels"] < 1:
            raise MicrophoneError("没有可用麦克风，请检查系统默认输入设备。")
    except Exception:
        raise MicrophoneError("未找到可用麦克风，请检查系统默认输入设备和麦克风权限。") from None

    chunks = queue.Queue(maxsize=300)  # At most 30 s of unprocessed audio.
    failed = []

    def capture(data, frames, timing, status):
        if stop_event.is_set() or failed:
            return
        if status.input_overflow:
            failed.append("麦克风音频丢失，请关闭占用设备的应用或检查系统负载后重试。")
            return
        try:
            chunks.put_nowait((bytes(data), int(time.monotonic() * 1000)))
        except queue.Full:
            failed.append("待识别音频已积压超过 30 秒，已停止采集。请缩短连续发言或降低系统负载后重试。")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
    try:
        stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16",
                                   blocksize=max(1, rate // 10), callback=capture)
        stream.start()
    except Exception:
        # Never expose driver messages or device identifiers in unexpected errors.
        if "stream" in locals():
            stream.close()
        raise MicrophoneError("无法打开麦克风，请检查默认输入设备、Windows 麦克风权限及设备占用。") from None

    last_level = 0.0
    device_name = str(info.get("name", "系统默认输入设备"))
    on_status(f"正在采集本机麦克风（{device_name}）")
    try:
        while not stop_event.is_set():
            if failed:
                raise MicrophoneError(failed[0])
            try:
                pcm, timestamp = chunks.get(timeout=0.1)
            except queue.Empty:
                if cloud:
                    for event in transcriber.poll():
                        on_transcript(event)
                        if copilot is not None and event.kind != "rejected":
                            copilot.submit_transcript(event.text, event.timestamp_ms, final=event.kind == "final", suspect=event.suspect)
                if not stream.active:
                    raise MicrophoneError("麦克风已断开，请检查输入设备后重新开始。")
                continue
            frame = av.AudioFrame.from_ndarray(np.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = rate
            for converted in resampler.resample(frame):
                output = converted.to_ndarray().astype("<i2", copy=False).tobytes()
                for event in transcriber.push_pcm(output, timestamp_ms=timestamp):
                    on_transcript(event)
                    if copilot is not None and event.kind != "rejected":
                        copilot.submit_transcript(event.text, event.timestamp_ms, final=event.kind == "final", suspect=event.suspect)
            if time.monotonic() - last_level >= 1:
                level, advice = input_quality(pcm)
                on_status(f"{device_name} · 输入音量 {level:.1%}" + (f" · {advice}" if advice else ""))
                last_level = time.monotonic()
    finally:
        stream.close()
