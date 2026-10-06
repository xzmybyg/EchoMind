param(
    [Parameter(Mandatory=$true)][string]$InstallDir,
    [Parameter(Mandatory=$true)][string]$StageDir,
    [Parameter(Mandatory=$true)][int]$ProcessId,
    [switch]$NoRestart
)
$ErrorActionPreference = 'Stop'
$install = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
$stage = [IO.Path]::GetFullPath($StageDir).TrimEnd('\')
$components = @('EchoMind.exe', '_internal', 'capture')
$token = [Guid]::NewGuid().ToString('N')
$backup = Join-Path $install ('.update-backup-' + $token)
$failed = Join-Path $install ('.update-failed-' + $token)
$ready = Join-Path $install ('.update-ready-' + $token)
$oldExited = $false
$saved = @()
$installed = @()
$statusFile = Join-Path (Split-Path -Parent $stage) 'result.json'
try {
    if ($install -eq $stage -or $install.StartsWith($stage + '\', [StringComparison]::OrdinalIgnoreCase) -or $stage.StartsWith($install + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Installation and staging directories must be separate.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $install 'EchoMind.exe') -PathType Leaf)) { throw 'Invalid installation directory.' }
    foreach ($directory in @($install, $stage)) {
        if ((Get-Item -LiteralPath $directory).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked directories are not supported.' }
    }
    foreach ($name in $components) {
        foreach ($directory in @($install, $stage)) {
            $candidate = [IO.Path]::GetFullPath((Join-Path $directory $name))
            if (-not $candidate.StartsWith($directory + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe component path.' }
            if (-not (Test-Path -LiteralPath $candidate)) { throw 'Missing application component.' }
            if ((Get-Item -LiteralPath $candidate).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked components are not supported.' }
        }
    }
    # Stage on the installation volume, so replacement moves are atomic even
    # when LOCALAPPDATA (download) and the application are on different drives.
    if (-not [IO.Path]::GetFullPath($ready).StartsWith($install + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe staging path.' }
    New-Item -ItemType Directory -Path $ready | Out-Null
    foreach ($name in $components) {
        Copy-Item -LiteralPath (Join-Path $stage $name) -Destination (Join-Path $ready $name) -Recurse
    }
    if ($ProcessId -gt 0) {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($process -and -not $process.WaitForExit(60000)) { throw 'EchoMind did not exit; old files were left unchanged.' }
    }
    $oldExited = $true
    New-Item -ItemType Directory -Path $backup | Out-Null
    foreach ($name in $components) {
        Move-Item -LiteralPath (Join-Path $install $name) -Destination (Join-Path $backup $name)
        $saved += $name
        Move-Item -LiteralPath (Join-Path $ready $name) -Destination (Join-Path $install $name)
        $installed += $name
    }
    if (-not $NoRestart) {
        # Relaunch the visible application, not a background helper window.
        $restarted = Start-Process -FilePath (Join-Path $install 'EchoMind.exe') -WorkingDirectory $install -PassThru
        if ($restarted.WaitForExit(5000)) { throw 'The new version exited immediately; restoring previous files.' }
    }
    @{status='complete'; backup=$backup} | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
} catch {
    $failure = $_.Exception.Message
    try {
        if ($installed.Count -gt 0) {
            New-Item -ItemType Directory -Path $failed | Out-Null
            foreach ($name in $installed) {
                Move-Item -LiteralPath (Join-Path $install $name) -Destination (Join-Path $failed $name)
            }
        }
        foreach ($name in $saved) {
            Move-Item -LiteralPath (Join-Path $backup $name) -Destination (Join-Path $install $name)
        }
        $state = 'rolled_back'
    } catch {
        $state = 'recovery_required'
        $failure += '; rollback failed: ' + $_.Exception.Message
    }
    @{status=$state; error=$failure; backup=$backup} | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
    if (-not $NoRestart) {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show("Update failed. Previous files are kept at: $backup`n$failure", 'EchoMind Update') | Out-Null
        if ($oldExited -and $state -eq 'rolled_back' -and (Test-Path -LiteralPath (Join-Path $install 'EchoMind.exe'))) {
            Start-Process -FilePath (Join-Path $install 'EchoMind.exe') -WorkingDirectory $install | Out-Null
        }
    }
    exit 1
}
