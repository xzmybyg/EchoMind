param([Parameter(Mandatory=$true)][string]$Manifest, [Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$spec = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
try {
    foreach ($case in $spec.cases) {
        for ($index = 0; $index -lt $case.parts.Count; $index++) {
            $part = $case.parts[$index]
            if ($part -isnot [string]) { continue }
            $fragmentIndex = 0
            foreach ($fragment in [regex]::Split($part, '([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*)*)')) {
                if (-not $fragment) { continue }
                $target = Join-Path $OutputDirectory ($case.id + '-' + $index + '-' + $fragmentIndex + '.wav')
                $fragmentIndex++
                if (Test-Path -LiteralPath $target) { continue }
                $voice = if ($fragment -match '^[A-Za-z]') { 'Microsoft Zira Desktop' } else { 'Microsoft Huihui Desktop' }
                $synth.SelectVoice($voice)
                $synth.SetOutputToWaveFile($target, $format)
                $synth.Speak($fragment)
                $synth.SetOutputToNull()
            }
        }
    }
} finally { $synth.Dispose() }
