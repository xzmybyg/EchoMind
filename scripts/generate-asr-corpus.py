"""Generate local TTS audio + independent references, without sending text/audio online."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave

import numpy as np

RATE = 16000


def read_wave(path):
    with wave.open(str(path), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (RATE, 1, 2)
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    # Remove synthesizer padding; preserve all explicitly authored pauses.
    active = np.flatnonzero(np.abs(samples) > 0.0003)
    return samples[max(0, active[0] - 800):min(len(samples), active[-1] + 800)] if active.size else samples


def write_wave(path, samples):
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        target.writeframes((samples * 32768).clip(-32768, 32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    spec_path = scripts / "asr-corpus.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory; an annotated corpus must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(scripts / "synthesize-asr-corpus.ps1"), "-Manifest", str(spec_path),
                    "-OutputDirectory", str(output / "tts")], check=True)
    rng = np.random.default_rng(20261010)
    chunks = [np.zeros(RATE * 2)]
    count = len(chunks[0])
    references = []
    for case in spec["cases"]:
        pieces = []
        for i, part in enumerate(case.get("parts", [])):
            if isinstance(part, str):
                fragments = sorted((output / "tts").glob(f"{case['id']}-{i}-*.wav"),
                                   key=lambda path: int(path.stem.rsplit("-", 1)[1]))
                pieces.extend(read_wave(path) for path in fragments)
            else:
                pieces.append(np.zeros(int(part * RATE / 1000)))
        audio = np.concatenate(pieces) if pieces else np.zeros(int(case["duration_ms"] * RATE / 1000))
        audio *= case.get("gain", 1)
        if "snr_db" in case:
            audio += rng.normal(0, np.sqrt(np.mean(audio ** 2)) / 10 ** (case["snr_db"] / 20), len(audio))
        if "noise_rms" in case:
            audio += rng.normal(0, case["noise_rms"], len(audio))
        if "hum_amplitude" in case:
            audio += case["hum_amplitude"] * np.sin(np.arange(len(audio)) * 2 * np.pi * 50 / RATE)
        filename = f"{case['id']}.wav"
        write_wave(output / filename, audio)
        references.append({**case, "reference": case.get("reference", "".join(p for p in case.get("parts", []) if isinstance(p, str))),
                           "start_ms": count * 1000 / RATE, "end_ms": (count + len(audio)) * 1000 / RATE,
                           "audio": filename})
        chunks.extend((audio, np.zeros(3 * RATE)))
        count += len(audio) + 3 * RATE
    write_wave(output / "corpus.wav", np.concatenate(chunks))
    manifest = {"description": spec["description"], "voices": ["Microsoft Huihui Desktop", "Microsoft Zira Desktop"],
                "sample_rate": RATE, "audio": "corpus.wav", "duration_seconds": count / RATE,
                "sha256": hashlib.sha256((output / "corpus.wav").read_bytes()).hexdigest(), "cases": references}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(references)} cases, {count / RATE:.1f} seconds: {output / 'corpus.wav'}")


if __name__ == "__main__":
    main()
