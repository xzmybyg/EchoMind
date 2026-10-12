"""Standalone JSON-line worker; executed by the separately installed Python."""
import base64
import contextlib
import json
import sys


def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.set_num_threads(4)
        processor = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            sys.argv[1], local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    send({"ready": True})
    for line in sys.stdin:
        request = json.loads(line)
        pcm = base64.b64decode(request["pcm"], validate=True)
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            inputs = processor.apply_transcription_request(audio=audio, language="Chinese", prompt="").to("cuda", torch.bfloat16)
            output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            tokens = output[:, inputs["input_ids"].shape[1]:]
            text = processor.decode(tokens, return_format="transcription_only")[0]
        send({"text": text, "truncated": tokens.shape[1] >= 160})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        send({"error": True})
        raise SystemExit(1)
