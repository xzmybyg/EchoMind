param(
    [ValidateSet('tiny', 'large-v3-turbo')]
    [string]$Model = 'large-v3-turbo'
)

$ErrorActionPreference = 'Stop'
$models = @{
    'tiny' = @{
        repo = 'Systran/faster-whisper-tiny'
        modelSize = 75538270
        modelSha256 = 'DCB76C6586FC06CBDAC6DD21F14CFD129CC4CDD9DCE19BF4FFA62E59CBE6E6D1'
        files = @('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt')
    }
    'large-v3-turbo' = @{
        repo = 'mobiuslabsgmbh/faster-whisper-large-v3-turbo'
        modelSize = 1617884929
        modelSha256 = 'E76620F83D5F5B69EFD3D87E3DC180C1BD21DF9FBEBACFD4335E5E1EFCC018DA'
        files = @('config.json', 'model.bin', 'preprocessor_config.json', 'tokenizer.json', 'vocabulary.json')
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot ".models\$Model"
New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null

foreach ($file in $models[$Model].files) {
    $destination = Join-Path $modelDirectory $file
    $uri = "https://huggingface.co/$($models[$Model].repo)/resolve/main/${file}?download=true"

    if ($file -eq 'model.bin') {
        if (-not (Test-Path -LiteralPath $destination)) {
            [IO.File]::Create($destination).Dispose()
        }
        $size = [long]$models[$Model].modelSize
        $position = (Get-Item -LiteralPath $destination).Length
        if ($position -gt $size) {
            throw "Model file is larger than expected: $destination"
        }
        $part = "$destination.part"
        while ($position -lt $size) {
            $lastByte = [Math]::Min($position + 33554432 - 1, $size - 1)
            $expectedRange = "bytes $position-$lastByte/$size"
            $valid = $false
            for ($attempt = 1; $attempt -le 3; $attempt++) {
                Write-Host "Downloading model.bin bytes $position-$lastByte of $size (attempt $attempt) ..."
                try {
                    $request = [Net.HttpWebRequest]::Create($uri)
                    $request.AddRange([long]$position, [long]$lastByte)
                    $response = $request.GetResponse()
                    $inputStream = $response.GetResponseStream()
                    $outputStream = [IO.File]::Create($part)
                    try { $inputStream.CopyTo($outputStream) }
                    finally { $outputStream.Dispose(); $inputStream.Dispose() }
                    $valid = [int]$response.StatusCode -eq 206 -and
                        $response.Headers['Content-Range'] -eq $expectedRange -and
                        (Get-Item -LiteralPath $part).Length -eq ($lastByte - $position + 1)
                    $response.Dispose()
                    if ($valid) { break }
                }
                catch {
                    if ($attempt -eq 3) { throw }
                }
                if ($attempt -lt 3) { Start-Sleep -Seconds 2 }
            }
            if (-not $valid) { throw "Invalid range response for $expectedRange" }
            $source = [IO.File]::OpenRead($part)
            $target = [IO.File]::Open($destination, [IO.FileMode]::Append, [IO.FileAccess]::Write)
            try { $source.CopyTo($target) }
            finally { $source.Dispose(); $target.Dispose() }
            $position = $lastByte + 1
        }
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash -ne $models[$Model].modelSha256) {
            throw "Model checksum mismatch: $destination"
        }
        Write-Host 'Model verified'
        continue
    }

    if (Test-Path -LiteralPath $destination) {
        Write-Host "Already have $file; skipping"
        continue
    }
    $temporary = "$destination.download"
    Write-Host "Downloading $file ..."
    Invoke-WebRequest -Uri $uri -OutFile $temporary -UseBasicParsing
    Move-Item -LiteralPath $temporary -Destination $destination
}

Write-Host "Model directory: $modelDirectory"
