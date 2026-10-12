"""Replay the 2026-10-09 microphone sample through three local engines.

The four reference sentences/time ranges below apply ONLY to this recording.
For other recordings, replace and verify these annotations before scoring.
No cloud inference or new recording is performed.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import wave
from statistics import median
from time import perf_counter

import numpy as np
from faster_whisper.audio import decode_audio

from echomind.asr import DEFAULT_HOTWORDS
from echomind.asr_backends import create_transcriber
from echomind.vad import SpeechVAD


def normalized(text):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text.lower())


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["whisper", "sensevoice", "paraformer", "qwen3-asr-1.7b"])
    parser.add_argument("--export-clips", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["PATH"] = str(root / ".runtime/cuda12") + os.pathsep + os.environ["PATH"]
    dll = os.add_dll_directory(str(root / ".runtime/cuda12")) if os.name == "nt" else None
    audio = decode_audio(str(args.audio), sampling_rate=16000)
    references = [(19000, 24000, "讲解一下React的Hooks", ["react", "hooks"]),
                  (24000, 31000, "讲解一下React的Hooks", ["react", "hooks"]),
                  (32000, 40000, "再说一下React和Vue的区别", ["react", "vue"]),
                  (41000, 53000, "然后再讲解一下ES6的Promise", ["es6", "promise"])]
    if args.export_clips:
        args.export_clips.mkdir(parents=True, exist_ok=True)
        cases = []
        for index, (start, end, reference, terms) in enumerate(references):
            filename = f"human-{index + 1}.wav"
            with wave.open(str(args.export_clips / filename), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes((audio[start * 16:end * 16] * 32768).clip(-32768, 32767).astype("<i2").tobytes())
            cases.append(dict(id=f"human-{index + 1}", audio=filename, reference=reference, terms=terms))
        (args.export_clips / "manifest.json").write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=2), encoding="utf-8")
    noise = np.random.default_rng(17).normal(0, 40 / 32768, 20 * 16000)
    hum = np.sin(np.arange(20 * 16000) * (2 * np.pi * 50 / 16000)) * (100 / 32768)
    full = np.concatenate([audio, np.zeros(20 * 16000), noise, hum])
    pcm = (full * 32768).clip(-32768, 32767).astype("<i2").tobytes()
    reports = []
    for engine, folder in [("whisper", "large-v3-turbo"), ("sensevoice", "sensevoice"), ("paraformer", "paraformer"), ("qwen3-asr-1.7b", "qwen3-asr-1.7b")]:
        if args.engine and engine != args.engine:
            continue
        started = perf_counter()
        stream = create_transcriber(engine=engine, model_size=str(root / ".models" / folder),
                                    device="cuda" if engine == "whisper" else "cpu", compute_type="int8_float16" if engine == "whisper" else "int8",
                                    vad=SpeechVAD(), context_window_ms=15000, hotwords=DEFAULT_HOTWORDS,
                                    retry_low_confidence=True)
        stream.warm_up()
        warm_seconds = perf_counter() - started
        events = []
        started = perf_counter()
        for offset in range(0, len(pcm), 3200):
            events.extend(asdict(e) for e in stream.push_pcm(pcm[offset:offset + 3200]))
        events.extend(asdict(e) for e in stream.flush())
        seconds = perf_counter() - started
        if hasattr(stream, "close"):
            stream.close()
        finals = [e for e in events if e["kind"] == "final"]
        checks = []
        # These four requests were the instructed/expected content of this sample.
        for start, end, reference, terms in references:
            text = " ".join(e["text"] for e in finals if start <= e["timestamp_ms"] < end)
            expected, actual = normalized(reference), normalized(text)
            checks.append(dict(reference=reference, transcript=text,
                               character_errors=edit_distance(expected, actual), reference_characters=len(expected),
                               terms={term: term in actual for term in terms}))
        report = dict(engine=engine, device="GPU" if engine in ("whisper", "qwen3-asr-1.7b") else "CPU / 4 threads",
                      warmup_seconds=warm_seconds, processing_seconds=seconds,
                      duration_seconds=len(full) / 16000, real_time_factor=seconds / (len(full) / 16000),
                      final_decode_median_ms=median([e["latency_ms"] for e in finals]) if finals else None,
                      background_events=[e for e in events if e["timestamp_ms"] > len(audio) / 16],
                      normalized_cer=sum(c["character_errors"] for c in checks) / sum(c["reference_characters"] for c in checks),
                      checks=checks, events=events)
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != "events"}, ensure_ascii=False, indent=2), flush=True)
        del stream
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    if dll:
        dll.close()


if __name__ == "__main__":
    main()
