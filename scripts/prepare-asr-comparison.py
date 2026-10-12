"""Download and verify selected local ASR models, without running inference."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "sensevoice": ("csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                   ("model.int8.onnx", "tokens.txt", "LICENSE")),
    "paraformer": ("csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en",
                   ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")),
    "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B-hf",
                        ("config.json", "generation_config.json", "model.safetensors", "processor_config.json",
                         "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md")),
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B-hf",
                        ("config.json", "generation_config.json", "model.safetensors", "processor_config.json",
                         "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md")),
}


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as stream:
        if "lfs" in entry:
            return hashlib.file_digest(stream, "sha256").hexdigest() == entry["lfs"]["sha256"]
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest() == entry["blobId"]


def prepare(name, spec):
    repo, filenames = spec
    # The same repository is mirrored for hosts unable to reach Hugging Face.
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
        base = "https://huggingface.co"
        try:
            response = client.get(f"{base}/api/models/{repo}?blobs=true", timeout=10)
            response.raise_for_status()
        except httpx.HTTPError:
            base = "https://hf-mirror.com"
            response = client.get(f"{base}/api/models/{repo}?blobs=true")
            response.raise_for_status()
        metadata = response.json()
        print(f"{name}: revision {metadata['sha']}", flush=True)
        files = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        folder = ROOT / ".models" / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            entry = files[filename]
            target = folder / filename
            if matches(target, entry):
                print(f"{name}/{filename}: reused", flush=True)
                continue
            if target.exists():
                raise RuntimeError(f"Existing file differs: {target}; not overwritten")
            part = target.with_suffix(target.suffix + ".part")
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry["size"]:
                raise RuntimeError(f"Invalid fragment: {part}")
            if offset < entry["size"]:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                url = f"{base}/{repo}/resolve/{metadata['sha']}/{filename}"
                with client.stream("GET", url, headers=headers) as download:
                    download.raise_for_status()
                    if offset and download.status_code != 206:
                        raise RuntimeError("Server did not resume fragment")
                    print(f"{name}/{filename}: downloading {entry['size'] / 1048576:.1f} MiB", flush=True)
                    with part.open("ab" if offset else "wb") as stream:
                        for chunk in download.iter_bytes(1024 * 1024):
                            stream.write(chunk)
            if not matches(part, entry):
                raise RuntimeError(f"Model digest mismatch: {part}")
            os.replace(part, target)
            print(f"{name}/{filename}: verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["sensevoice", "paraformer"])
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, name, MODELS[name]) for name in args.models]
        for future in futures:
            future.result()
