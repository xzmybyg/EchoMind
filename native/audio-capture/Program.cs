using System.Diagnostics;
using NAudio.CoreAudioApi;
using NAudio.Wave;

if (args.Length != 2 || args[0] != "--pid" || !uint.TryParse(args[1], out uint pid) || pid == 0 || pid > int.MaxValue)
{
    Console.Error.WriteLine("Usage: EchoMind.Capture --pid <process-id> > capture.pcm");
    return 2;
}

if (!OperatingSystem.IsWindowsVersionAtLeast(10, 0, 20348))
{
    Console.Error.WriteLine("Process loopback requires Windows build 20348 or later.");
    return 2;
}

try
{
    using var process = Process.GetProcessById((int)pid);
    await using var recorder = await new WasapiRecorderBuilder()
        .WithProcessLoopback(pid, ProcessLoopbackMode.IncludeTargetProcessTree)
        .WithFormat(new WaveFormat(16000, 16, 1))
        .WithBufferLength(30)
        .BuildAsync();

    using var quit = new ManualResetEvent(false);
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; quit.Set(); };
    var output = Console.OpenStandardOutput();
    var silence = new byte[3200]; // 100 ms at 16 kHz, mono, 16-bit.
    var writeLock = new object();
    long lastPacket = Stopwatch.GetTimestamp();
    Exception? captureError = null;
    bool running = true;

    recorder.DataAvailable += (data, _, _, _) =>
    {
        lock (writeLock)
        {
            output.Write(data);
            lastPacket = Stopwatch.GetTimestamp();
        }
    };
    recorder.RecordingStopped += (_, e) =>
    {
        captureError = e.Exception;
        quit.Set();
    };

    recorder.StartRecording();
    Console.Error.WriteLine($"Capturing PID {pid} and its children: 16000 Hz, mono, signed 16-bit PCM on stdout.");
    using var silenceTimer = new Timer(_ =>
    {
        try
        {
            lock (writeLock)
            {
                if (running && Stopwatch.GetElapsedTime(lastPacket) >= TimeSpan.FromMilliseconds(100))
                {
                    output.Write(silence);
                    lastPacket = Stopwatch.GetTimestamp();
                }
            }
        }
        catch (Exception ex) { captureError = ex; quit.Set(); }
    }, null, 100, 100);

    while (!quit.WaitOne(200))
    {
        if (process.HasExited)
            throw new InvalidOperationException("Target process exited; select its new PID to restart capture.");
    }

    lock (writeLock) { running = false; }
    silenceTimer.Change(Timeout.Infinite, Timeout.Infinite);
    recorder.StopRecording();
    output.Flush();
    if (captureError != null) throw captureError;
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Capture failed: {ex.Message} (0x{ex.HResult:X8})");
    return 1;
}
