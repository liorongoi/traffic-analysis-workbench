# Sets up LanePulse training on a Windows computer with an NVIDIA GPU. No administrator rights needed.
# Run it from the training kit folder:
#   powershell -ExecutionPolicy Bypass -File setup-training-windows.ps1
# It needs the internet once (Python packages and the starting weights), so use a connected computer,
# not the offline workstation. Everything goes into %USERPROFILE%\LanePulse-training.
param([string]$Dir = (Join-Path $env:USERPROFILE "LanePulse-training"))
$ErrorActionPreference = "Stop"
$Kit = $PSScriptRoot
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

# Python 3.11 or later: the py launcher, python on PATH, or one fetched by uv into the user's folder.
$Py = ""
foreach ($v in "3.13", "3.12", "3.11") {
  try { $Py = (& py "-$v" -c "import sys; print(sys.executable)" 2>$null).Trim() } catch { $Py = "" }
  if ($Py) { break }
}
if (-not $Py) {
  try {
    $ok = & python -c "import sys; print(sys.version_info >= (3, 11))" 2>$null
    if ($ok -eq "True") { $Py = (& python -c "import sys; print(sys.executable)").Trim() }
  } catch { $Py = "" }
}
if (-not $Py) {
  Write-Host "Getting Python 3.12 (into your user folder, no administrator rights)..."
  if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
  }
  uv python install 3.12
  $Py = (uv python find 3.12).Trim()
}
Write-Host "Python: $Py"

& $Py -m venv (Join-Path $Dir "venv")
$VPy = Join-Path $Dir "venv\Scripts\python.exe"
$ErrorActionPreference = "Continue"
& $VPy -m pip install --quiet --upgrade pip
# PyTorch with CUDA when an NVIDIA driver is present (CUDA 12.8 builds need driver 570 or later), else the CPU build.
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$index = if ($hasNvidia) { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
Write-Host "Installing PyTorch ($(if ($hasNvidia) { 'NVIDIA CUDA' } else { 'CPU only: no NVIDIA driver found' })), a few minutes..."
& $VPy -m pip install --quiet torch --index-url $index
if ($LASTEXITCODE -ne 0) { throw "PyTorch could not be installed." }
$Whl = Get-ChildItem $Kit -Filter "trafficwb-*.whl" | Sort-Object Name | Select-Object -Last 1
& $VPy -m pip install --quiet --upgrade "$($Whl.FullName)[train]"
if ($LASTEXITCODE -ne 0) { throw "LanePulse training could not be installed." }
$ErrorActionPreference = "Stop"

Copy-Item (Join-Path $Kit "TRAINING.txt") (Join-Path $Dir "TRAINING.txt") -Force -ErrorAction SilentlyContinue
@"
@echo off
rem Trains the LanePulse vehicle detector. Drag the exported dataset folder onto this file, or run it and paste the path.
set DS=%~1
if "%DS%"=="" set /p DS=Exported dataset folder: 
set /p EP=Rounds of training (Enter for 40): 
if "%EP%"=="" set EP=40
"$Dir\venv\Scripts\trafficwb-train.exe" run --dataset "%DS%" --out "$Dir\models" --epochs %EP%
echo.
echo Copy the newest .zip in $Dir\models to the LanePulse computer and drag it onto import-model.cmd there.
explorer "$Dir\models"
pause
"@ | Set-Content -Encoding ASCII (Join-Path $Dir "train.cmd")

& (Join-Path $Dir "venv\Scripts\trafficwb-train.exe") device
Write-Host ""
Write-Host "Ready. To train, drag an exported dataset folder onto $Dir\train.cmd"
