# Offline install or update of the Traffic Analysis Workbench on a Windows workstation.
# Run from the unpacked bundle folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Target C:\TrafficWorkbench -Data D:\TrafficData -BackupDir E:\TrafficBackups
# Nothing is downloaded. Each version gets its own folder, so the previous one stays for rollback.
param(
  [string]$Target = "C:\TrafficWorkbench",
  [string]$Data = "C:\TrafficWorkbench\data",
  [string]$BackupDir = "",
  [string]$Python = "",
  [int]$Port = 8765,
  [string]$Ort = "",
  [ValidateSet("signin", "boot", "none")]
  [string]$AutoStart = "signin",   # signin needs no administrator rights; boot is for an IT administrator
  [switch]$SkipCheck
)
$ErrorActionPreference = "Stop"
$Bundle = $PSScriptRoot
$Info = Get-Content (Join-Path $Bundle "bundle.json") -Raw | ConvertFrom-Json
Write-Host "Traffic Analysis Workbench $($Info.version) for Python $($Info.python)"

# 1. Check every file against SHA256SUMS.
Get-Content (Join-Path $Bundle "SHA256SUMS") | ForEach-Object {
  $parts = $_ -split "  ", 2
  if ($parts.Count -eq 2) {
    $f = Join-Path $Bundle $parts[1]
    $h = (Get-FileHash -Algorithm SHA256 $f).Hash.ToLower()
    if ($h -ne $parts[0]) { throw "Checksum mismatch: $($parts[1]). Copy the bundle again." }
  }
}
Write-Host "Bundle checksums verified."

# 2. Python: given, the py launcher, or the installer shipped in the bundle.
if (-not $Python) {
  try { $Python = (& py "-$($Info.python)" -c "import sys; print(sys.executable)").Trim() } catch { $Python = "" }
}
if (-not $Python) {
  try {
    $v = (& python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
    if ($v -eq $Info.python) { $Python = (& python -c "import sys; print(sys.executable)").Trim() }
  } catch { }
}
if (-not $Python) {
  $installer = Get-ChildItem $Bundle -Filter "python-*.exe" | Select-Object -First 1
  if (-not $installer) { throw "Python $($Info.python) is not installed and no python-*.exe is in the bundle." }
  $pyDir = Join-Path $Target "python$($Info.python -replace '\.','')"
  Start-Process -Wait -FilePath $installer.FullName -ArgumentList "/quiet","InstallAllUsers=0","PrependPath=0","Include_test=0","TargetDir=$pyDir"
  $Python = Join-Path $pyDir "python.exe"
}
Write-Host "Python: $Python"

# 3. Wheels collect in one local wheelhouse, so a small update bundle (application only) finds the runtime there.
New-Item -ItemType Directory -Force -Path $Target | Out-Null
$Venv = Join-Path $Target "app-$($Info.version)"
if (Test-Path $Venv) { throw "$Venv already exists. Remove it first to reinstall the same version." }
$Wheels = Join-Path $Target "wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheels | Out-Null
Get-ChildItem (Join-Path $Bundle "wheelhouse") -Filter *.whl | ForEach-Object {
  $dest = Join-Path $Wheels $_.Name
  if (-not (Test-Path $dest)) { Copy-Item $_.FullName $dest }
}

# 4. Model packs: add the bundle's versions; existing versions are kept.
$Models = Join-Path $Target "models"
$BundleModels = Join-Path $Bundle "models"
if (Test-Path $BundleModels) { Get-ChildItem $BundleModels -Directory | ForEach-Object {
  Get-ChildItem $_.FullName -Directory | ForEach-Object {
    $dest = Join-Path $Models (Join-Path $_.Parent.Name $_.Name)
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null; Copy-Item -Recurse $_.FullName $dest }
  }
} }

# 5. Which ONNX Runtime build: CUDA when the NVIDIA driver is new enough for the bundled CUDA libraries, else DirectML.
function Get-NvidiaDriver {
  try {
    $out = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { return [version](($out | Select-Object -First 1).Trim()) }
  } catch { }
  return $null
}
$Choice = $Ort
if (-not $Choice) { $Choice = $Info.ort }
if ($Choice -eq "auto") {
  $drv = Get-NvidiaDriver
  $need = [version]$Info.cuda_min_driver
  if ($drv -and $drv -ge $need) { $Choice = "cuda"; Write-Host "NVIDIA driver $drv (needs $need for CUDA): using CUDA." }
  elseif ($drv) { $Choice = "directml"; Write-Host "NVIDIA driver $drv is older than ${need}: using DirectML." }
  else { $Choice = "directml"; Write-Host "No NVIDIA driver found: using DirectML (falls back to CPU without a GPU)." }
}

# Install into a fresh environment and check it end to end with no network, before switching to it.
function Install-Env([string]$build) {
  & $Python -m venv $Venv | Out-Host
  $script:VPy = Join-Path $Venv "Scripts\python.exe"
  & $script:VPy -m pip install --no-index --find-links $Wheels "trafficwb[probe,$build]==$($Info.version)" | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "pip install failed." }
  if ($SkipCheck) { return "unchecked" }
  $work = Join-Path $env:TEMP ("twb-check-" + [guid]::NewGuid())
  $lines = & $script:VPy (Join-Path $Bundle "offline_smoke.py") --work $work --models $Models
  $code = $LASTEXITCODE
  $lines | ForEach-Object { Write-Host $_ }
  if ($code -ne 0) { throw "The install check failed; the previous version stays in use." }
  $result = ($lines | Select-Object -Last 1) | ConvertFrom-Json
  return $result.accelerator.accelerator
}
$used = Install-Env $Choice
if ($Choice -eq "cuda" -and $used -ne "cuda" -and -not $SkipCheck -and ($Info.wheels | Where-Object { $_ -like "onnxruntime_directml-*" })) {
  Write-Host "CUDA did not start on this machine (models ran on $used); installing the DirectML build instead."
  Remove-Item -Recurse -Force $Venv
  $Choice = "directml"
  $used = Install-Env $Choice
}
Write-Host "ONNX Runtime build: ${Choice}; models run on: ${used}"

# 6. Back up the data before the new version touches it, then switch the start script.
New-Item -ItemType Directory -Force -Path $Data | Out-Null
if ($BackupDir -and (Test-Path (Join-Path $Data "control.db"))) {
  $prev = Join-Path $Target "current.txt"
  if (Test-Path $prev) {
    $prevVenv = (Get-Content $prev -Raw).Trim()
    & (Join-Path $prevVenv "Scripts\trafficwb-admin.exe") backup --data $Data --to $BackupDir
    if ($LASTEXITCODE -ne 0) { throw "Backup before the update failed; nothing was switched." }
  }
}
$backupArgs = ""
if ($BackupDir) { $backupArgs = "--backup-dir `"$BackupDir`"" }
@"
@echo off
rem Starts the Traffic Analysis Workbench $($Info.version). Open http://127.0.0.1:$Port in a browser.
set TWB_MODELS=$Models
"$Venv\Scripts\trafficwb-web.exe" serve --data "$Data" --port $Port $backupArgs
"@ | Set-Content -Encoding ASCII (Join-Path $Target "start.cmd")
if (Test-Path (Join-Path $Target "current.txt")) { Copy-Item (Join-Path $Target "current.txt") (Join-Path $Target "previous.txt") -Force }
Set-Content -Path (Join-Path $Target "current.txt") -Value $Venv
Copy-Item (Join-Path $Bundle "autostart.ps1") (Join-Path $Target "autostart.ps1") -Force
Write-Host "Installed $($Info.version)."

# 7. Automatic start. An update restarts a server that already starts by itself, so it runs the new version.
$auto = Join-Path $Target "autostart.ps1"
$hasLink = Test-Path (Join-Path ([Environment]::GetFolderPath("Startup")) "Traffic Analysis Workbench.lnk")
$hasTask = [bool](Get-ScheduledTask -TaskName "Traffic Analysis Workbench" -ErrorAction SilentlyContinue)
if ($hasLink -or $hasTask) {
  & $auto -Mode restart -Target $Target -Port $Port
} elseif ($AutoStart -ne "none") {
  & $auto -Mode $AutoStart -Target $Target -Port $Port
  if ($LASTEXITCODE -eq 2) { & $auto -Mode signin -Target $Target -Port $Port }
} else {
  Write-Host "Start with $Target\start.cmd."
}
Write-Host "Open http://127.0.0.1:$Port in a browser. The first account is created there, on this computer."
