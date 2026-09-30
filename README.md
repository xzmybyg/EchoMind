# EchoMind

EchoMind is a Windows conversation assistant. The first MVP captures audio from a selected Tencent Meeting process and shows live Chinese captions in a terminal. Question detection, answers, knowledge retrieval, and the desktop overlay are later milestones.

## MVP 1 setup

Requirements: Windows 10 build 20348 or newer, .NET 9 SDK, Python 3.11, and an NVIDIA GPU for the recommended CUDA path. Capturing a meeting requires consent from its participants. Audio is processed in memory; this MVP does not save recordings or transcripts.

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\services\transcription[dev]"
dotnet build .\native\audio-capture\EchoMind.Capture.csproj -c Release
.\scripts\download-model.ps1 -Model large-v3-turbo
.\scripts\install-cuda-runtime.ps1
.\scripts\run-captions.ps1
```

The last command lists Tencent Meeting processes. Select the process producing remote audio, or pass `-TargetPid 12345`. For a quick pipeline smoke test, download the smaller `tiny` model and run `run-captions.ps1 -Model tiny`; it is not intended for production Chinese captions. Use `-Device cpu` if CUDA is unavailable (the CUDA runtime step can then be skipped).

The model download script uses official Hugging Face model files and stores them under the ignored `.models/` directory. The CUDA script downloads the Windows CUDA 12/cuDNN 9 libraries linked by [faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu), checks the archive hash, and stores them under the ignored `.runtime/` directory. The first model download is large and may take time.

Run logic tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests services\transcription\tests
```

This version uses Windows application process loopback: it captures the selected PID and its child processes, not the entire system output. Tencent Meeting may play audio in a child or separate process; if captions stay silent, select the actual audio process. Partial captions can change before they become final. The current lightweight energy gate may miss quiet speech; replacing it with Silero VAD is a follow-up after baseline measurements.

## Layout

- `native/audio-capture/`: Windows process loopback to 16 kHz mono PCM.
- `services/transcription/`: Chinese faster-whisper captions and process-selection CLI.
- `scripts/`: local model and runtime setup, plus the launch command.
