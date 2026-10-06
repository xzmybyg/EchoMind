"""Default microphone input; capture callback never runs Whisper or network IO."""

import queue
import time


class MicrophoneError(RuntimeError):
    pass


def run_microphone(model, device, compute_type, copilot=None, *, stop_event, on_transcript, on_status):
    import av
    import numpy as np
    import sounddevice as sd
    from .asr import Transcriber

    transcriber = Transcriber(model_size=model, device=device, compute_type=compute_type)
    on_status("正在预热识别模型…")
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
    on_status("正在采集本机麦克风（系统默认输入设备）")
    try:
        while not stop_event.is_set():
            if failed:
                raise MicrophoneError(failed[0])
            try:
                pcm, timestamp = chunks.get(timeout=0.1)
            except queue.Empty:
                if not stream.active:
                    raise MicrophoneError("麦克风已断开，请检查输入设备后重新开始。")
                continue
            frame = av.AudioFrame.from_ndarray(np.frombuffer(pcm, dtype="<i2").reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = rate
            for converted in resampler.resample(frame):
                output = converted.to_ndarray().astype("<i2", copy=False).tobytes()
                for event in transcriber.push_pcm(output, timestamp_ms=timestamp):
                    on_transcript(event)
                    if copilot is not None:
                        copilot.submit_transcript(event.text, event.timestamp_ms, final=event.kind == "final", suspect=event.suspect)
            if time.monotonic() - last_level >= 1:
                level = float(np.sqrt(np.mean(np.frombuffer(pcm, dtype="<i2").astype(np.float32) ** 2))) / 32768
                on_status(f"本机麦克风 · 输入音量 {level:.1%}")
                last_level = time.monotonic()
    finally:
        stream.close()
