"""Replay an independently annotated corpus through the actual local ASR pipeline."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import time
import wave

from echomind.asr import DEFAULT_HOTWORDS
from echomind.asr_backends import create_transcriber
from echomind.question import QuestionBuffer
from echomind.vad import SpeechVAD


def normalized(text):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text.lower())


def distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        nxt = [i]
        for j, b in enumerate(right, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--engine", choices=["whisper", "sensevoice", "paraformer", "qwen3-asr-1.7b"], default="whisper")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--no-hotwords", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with (args.manifest.parent / manifest["audio"]).open("rb") as source:
        if manifest.get("sha256") and hashlib.file_digest(source, "sha256").hexdigest() != manifest["sha256"]:
            raise ValueError("Corpus audio differs from its annotated manifest")
    folder = "large-v3-turbo" if args.engine == "whisper" else args.engine
    cuda = root / ".runtime/cuda12"
    os.environ["PATH"] = str(cuda) + os.pathsep + os.environ.get("PATH", "")
    dll = os.add_dll_directory(str(cuda)) if os.name == "nt" else None
    device = "cuda" if args.engine == "whisper" else "cpu"
    transcriber = create_transcriber(engine=args.engine, model_size=str(root / ".models" / folder),
                                    device=device, compute_type="int8_float16" if device == "cuda" else "int8",
                                    vad=SpeechVAD(), context_window_ms=15000,
                                    retry_low_confidence=args.retry,
                                    hotwords=None if args.no_hotwords else DEFAULT_HOTWORDS)
    before = time.perf_counter()
    transcriber.warm_up()
    warmup = time.perf_counter() - before
    events, questions = [], []
    buffer = QuestionBuffer()
    started = time.perf_counter()
    with wave.open(str(args.manifest.parent / manifest["audio"]), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
        samples = 0
        while pcm := source.readframes(1600):
            samples += len(pcm) // 2
            if args.realtime:
                time.sleep(max(0, started + samples / 16000 - time.perf_counter()))
            emitted = transcriber.push_pcm(pcm)
            observed = (time.perf_counter() - started) * 1000
            for event in emitted:
                events.append({**asdict(event), "observed_ms": observed})
                if event.kind == "final" and not event.suspect:
                    question = buffer.push(event.text, event.timestamp_ms)
                    if question:
                        questions.append(asdict(question))
        events.extend(asdict(e) for e in transcriber.flush())
    seconds = time.perf_counter() - started
    if hasattr(transcriber, "close"):
        transcriber.close()
    checks = []
    for case in manifest["cases"]:
        selected = [e for e in events if case["start_ms"] <= e["timestamp_ms"] < case["end_ms"] + 3000]
        finals = [e for e in selected if e["kind"] == "final"]
        text = " ".join(e["text"] for e in finals)
        expected, actual = normalized(case["reference"]), normalized(text)
        checks.append(dict(id=case["id"], reference=case["reference"], transcript=text,
                           character_errors=distance(expected, actual), reference_characters=len(expected),
                           terms={t: normalized(t) in actual for t in case.get("terms", [])},
                           final_count=len(finals), rejected_count=sum(e["kind"] == "rejected" for e in selected),
                           suspect_count=sum(e["suspect"] for e in finals),
                           background_events=selected if not expected else [],
                           questions=[q["text"] for q in questions if case["start_ms"] <= q["timestamp_ms"] < case["end_ms"] + 3000],
                           final_delay_ms=(finals[-1].get("observed_ms", finals[-1]["timestamp_ms"]) - case["end_ms"])
                           if finals and args.realtime else None))
    report = dict(engine=args.engine, audio_sha256=manifest.get("sha256"), realtime=args.realtime,
                  retry=args.retry, hotwords=not args.no_hotwords,
                  question_evaluation="Local rule candidates only, excluding suspect text; not the Jev/LLM answer pipeline",
                  warmup_seconds=warmup, processing_seconds=seconds, duration_seconds=samples / 16000,
                  final_decode_median_ms=median([e["latency_ms"] for e in events if e["kind"] == "final"])
                  if any(e["kind"] == "final" for e in events) else None,
                  character_errors=sum(c["character_errors"] for c in checks),
                  reference_characters=sum(c["reference_characters"] for c in checks),
                  terms_hit=sum(sum(c["terms"].values()) for c in checks),
                  terms_total=sum(len(c["terms"]) for c in checks),
                  background_events=sum(len(c["background_events"]) for c in checks), checks=checks, events=events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("events", "checks")}, ensure_ascii=False), flush=True)
    for case in checks:
        print(json.dumps(case, ensure_ascii=False), flush=True)
    if dll:
        dll.close()


if __name__ == "__main__":
    main()
