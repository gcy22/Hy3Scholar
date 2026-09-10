param(
    [Parameter(Mandatory = $true)]
    [string]$DemoDirectory
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$narrationPath = Join-Path $DemoDirectory "narration.json"
$audioDirectory = Join-Path $DemoDirectory "build\audio"
New-Item -ItemType Directory -Force -Path $audioDirectory | Out-Null
$segments = Get-Content -LiteralPath $narrationPath -Raw -Encoding UTF8 | ConvertFrom-Json

$voiceName = "Microsoft Huihui Desktop"
$availableVoices = @(
    (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() |
        ForEach-Object { $_.VoiceInfo.Name }
)
if ($voiceName -notin $availableVoices) {
    $voiceName = $availableVoices | Where-Object { $_ -match "Huihui|Kangkang|Yaoyao" } | Select-Object -First 1
}
if (-not $voiceName) {
    throw "没有找到可用的中文系统语音"
}

foreach ($segment in $segments) {
    $speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $speaker.SelectVoice($voiceName)
    $speaker.Rate = 3
    $speaker.Volume = 100
    $fileName = "{0:D2}.wav" -f [int]$segment.id
    $outputPath = Join-Path $audioDirectory $fileName
    $speaker.SetOutputToWaveFile($outputPath)
    $speaker.Speak([string]$segment.narration)
    $speaker.SetOutputToNull()
    $speaker.Dispose()
    Write-Output "Narration $fileName generated"
}

Write-Output "Chinese voice: $voiceName"
