# Process audio capture

Build on Windows 10 build 20348+ with the .NET 9 SDK:

```powershell
dotnet build native/audio-capture/EchoMind.Capture.csproj -c Release
```

Run against the process that renders Tencent Meeting's remote audio:

```powershell
native/audio-capture/bin/Release/net9.0/EchoMind.Capture.exe --pid 12345 > meeting.pcm
```

Standard output is raw 16 kHz mono signed 16-bit little-endian PCM, without a WAV header. Diagnostics go to standard error. The capture includes the selected process and its child processes, but no other applications. It writes zero samples in 100 ms blocks while Windows delivers no audio packets so downstream VAD can detect pauses. Ctrl+C stops capture. When the target PID exits, the process reports an error and stops; choose the new PID and restart.

The virtual process-loopback device is not tied to a physical output device. Tencent Meeting may render audio in a different process from its main window; select the audio-rendering PID or its ancestor. This module does not capture the local microphone.
