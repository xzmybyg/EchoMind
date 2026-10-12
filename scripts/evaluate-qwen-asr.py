"""Evaluate native Qwen3-ASR on local corpus clips in an isolated environment."""
import argparse
import json
import os
from pathlib import Path
import time
import wave

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-hotwords", action="store_true")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    torch.set_num_threads(4)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(str(args.model), local_files_only=True,
                                                   dtype=dtype).to(args.device).eval()
    print(f"Loaded in {time.perf_counter() - started:.2f}s", flush=True)
    results = []
    for case in manifest["cases"]:
        with wave.open(str(args.manifest.parent / case["audio"]), "rb") as source:
            assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
            audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        before = time.perf_counter()
        inputs = processor.apply_transcription_request(audio=audio, language="Chinese",
                    prompt="" if args.no_hotwords else "React Hooks, React Fiber, Vue, JavaScript, TypeScript, ES6, Promise").to(args.device, dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = processor.decode(output[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
        result = dict(id=case["id"], reference=case["reference"], transcript=text,
                      audio_seconds=len(audio) / 16000,
                      processing_ms=(time.perf_counter() - before) * 1000,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576 if args.device == "cuda" else None,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1048576 if args.device == "cuda" else None)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"engine": args.model.name, "model_path": str(args.model.resolve()),
                              "torch_version": torch.__version__, "device": args.device,
                              "hotwords": not args.no_hotwords, "audio_sha256": manifest.get("sha256"),
                              "note": "Whole pre-segmented clips; no streaming/VAD. First clip includes warmup.",
                              "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
