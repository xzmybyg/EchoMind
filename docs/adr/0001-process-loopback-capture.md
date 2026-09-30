# ADR 0001: MVP 1 process loopback capture

Status: accepted for MVP 1 (2026-10-01)

EchoMind needs audio from a chosen meeting application without recording unrelated system output. Windows Application Loopback captures a target PID and its child processes. The current development machine supports the required Windows API.

For the first working version, the capture executable uses .NET 9 and NAudio.Wasapi 3.0. It writes 16 kHz mono signed 16-bit PCM to standard output; the Python transcription process reads that stream. This keeps the Windows callback and COM handling inside a maintained library while the product pipeline is still being validated. The original Rust/WASAPI direction remains an option once process selection and latency have been measured against Tencent Meeting.

Tradeoffs:

- The .NET 9 runtime and NAudio package are additional installation requirements.
- If Tencent Meeting renders audio in a separate process outside the selected process tree, the user must select that audio process.
- The PCM stream has no timestamp header. Current timing logs measure recognition time and the interval from Python receiving a chunk to displaying text; they are not a calibrated end-to-end audio latency measurement.

The capture boundary is intentionally small: `--pid` in, raw PCM on stdout, diagnostics on stderr. A later native implementation can replace it without changing the ASR interface.
