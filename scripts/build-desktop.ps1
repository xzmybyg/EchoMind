param(
    [string]$OutputDirectory = '',
    [switch]$IncludeComponents
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$model = Join-Path $repo '.models\large-v3-turbo'
$cuda = Join-Path $repo '.runtime\cuda12'
$scratch = Join-Path $repo '.runtime\desktop-build'
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo 'dist' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$package = Join-Path $OutputDirectory 'EchoMind'

if (Test-Path -LiteralPath $package) {
    throw "Package already exists: $package. Choose a fresh -OutputDirectory; existing packages are never deleted."
}
$inputs = @($python)
if ($IncludeComponents) { $inputs += @((Join-Path $model 'model.bin'), (Join-Path $cuda 'cublas64_12.dll')) }
foreach ($required in $inputs) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required build input not found: $required" }
}
if (-not (Test-Path -LiteralPath (Join-Path $repo 'services\transcription\echomind\gui.py'))) {
    throw 'The desktop UI module is missing.'
}
New-Item -ItemType Directory -Force -Path $scratch, $OutputDirectory | Out-Null

& $python -m PyInstaller --version
if ($LASTEXITCODE -ne 0) { throw 'Install the desktop-build extra in the project virtual environment first.' }
$arguments = @(
    '-m', 'PyInstaller', '--onedir', '--windowed', '--noupx', '--name', 'EchoMind',
    '--paths', (Join-Path $repo 'services\transcription'),
    '--distpath', $OutputDirectory, '--workpath', (Join-Path $scratch 'work'),
    '--specpath', $scratch, '--exclude-module', 'pytest',
    '--collect-all', 'faster_whisper', '--collect-all', 'ctranslate2',
    '--collect-all', 'av', '--collect-all', 'tokenizers',
    '--collect-all', 'sounddevice', '--collect-all', '_sounddevice_data',
    '--collect-all', 'opencc',
    '--add-data', ((Join-Path $repo 'services\transcription\echomind\windows_ocr.ps1') + ';echomind'),
    '--add-data', ((Join-Path $repo 'services\transcription\echomind\windows_update.ps1') + ';echomind'),
    '--copy-metadata', 'faster-whisper', '--copy-metadata', 'ctranslate2',
    '--copy-metadata', 'av', '--copy-metadata', 'tokenizers',
    (Join-Path $PSScriptRoot 'desktop-entry.pyw')
)
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Windowed executable build failed.' }

& dotnet publish (Join-Path $repo 'native\audio-capture\EchoMind.Capture.csproj') -c Release -r win-x64 --self-contained true -o (Join-Path $package 'capture')
if ($LASTEXITCODE -ne 0) { throw 'Audio capture publish failed.' }

if ($IncludeComponents) {
    $modelOutput = Join-Path $package 'models\large-v3-turbo'
    $cudaOutput = Join-Path $package 'cuda'
    New-Item -ItemType Directory -Force -Path $modelOutput, $cudaOutput | Out-Null
    # Do not distribute incomplete download fragments or any local settings/secrets.
    Get-ChildItem -LiteralPath $model -File | Where-Object { $_.Extension -ne '.part' } | Copy-Item -Destination $modelOutput
    Get-ChildItem -LiteralPath $cuda -Filter '*.dll' -File | Copy-Item -Destination $cudaOutput
}
Write-Host "Desktop package ready: $(Join-Path $package 'EchoMind.exe')"
Write-Host 'Distribute the entire EchoMind folder; the exe needs its adjacent runtime. Missing model/GPU components are downloaded on first launch.'
