$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDirectory = Join-Path $projectRoot '.runtime'
$archive = Join-Path $runtimeDirectory 'cuda12-libs.7z'
$libraries = Join-Path $runtimeDirectory 'cuda12'
$expectedHash = '89D396373E2781E01FDD58D35A73AADF9B2DBA83D3DCD05A838B9115D50427C3'
$uri = 'https://github.com/Purfview/whisper-standalone-win/releases/download/libs/cuBLAS.and.cuDNN_CUDA12_win_v2.7z'

New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $uri -OutFile $archive -UseBasicParsing
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash -ne $expectedHash) {
    throw "CUDA runtime archive checksum mismatch: $archive"
}

New-Item -ItemType Directory -Force -Path $libraries | Out-Null
tar -xf $archive -C $libraries
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $libraries 'cublas64_12.dll'))) {
    throw 'CUDA runtime extraction failed'
}
Write-Host "CUDA runtime ready: $libraries"
