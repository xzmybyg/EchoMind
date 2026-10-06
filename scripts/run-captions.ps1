param(
    [int]$TargetPid = 0,
    [ValidateSet('tiny', 'large-v3-turbo')]
    [string]$Model = 'large-v3-turbo',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [ValidateSet('off', 'demo', 'live')]
    [string]$Answers = 'off',
    [ValidateSet('rules', 'jev')]
    [string]$QuestionGate = 'rules'
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = 'utf-8'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$capture = Join-Path $projectRoot 'native\audio-capture\bin\Release\net9.0\EchoMind.Capture.exe'
$modelDirectory = Join-Path $projectRoot ".models\$Model"
$cudaDirectory = Join-Path $projectRoot '.runtime\cuda12'

foreach ($required in @($python, $capture, (Join-Path $modelDirectory 'model.bin'))) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing prerequisite: $required"
    }
}

if ($Device -eq 'cuda') {
    $cublas = Join-Path $cudaDirectory 'cublas64_12.dll'
    if (-not (Test-Path -LiteralPath $cublas)) {
        throw "CUDA runtime missing: $cublas"
    }
    $env:PATH = "$cudaDirectory;$env:PATH"
}

$precision = if ($Device -eq 'cuda') { 'int8_float16' } else { 'int8' }
$arguments = @('-m', 'echomind.cli', '--capture-exe', $capture,
    '--model', $modelDirectory, '--device', $Device, '--compute-type', $precision,
    '--answers', $Answers, '--question-gate', $QuestionGate)
if ($TargetPid -gt 0) {
    $arguments += @('--pid', "$TargetPid")
}

& $python @arguments
exit $LASTEXITCODE
