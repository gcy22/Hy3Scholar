$ErrorActionPreference = "Stop"
$demoDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $demoDirectory
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

$env:PYTHONUTF8 = "1"
& $python (Join-Path $demoDirectory "create_slides.py")
& (Join-Path $demoDirectory "generate_audio.ps1") -DemoDirectory $demoDirectory
& $python (Join-Path $demoDirectory "build_video.py")
