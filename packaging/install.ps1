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

# 3. A fresh environment for this version, next to earlier ones.
New-Item -ItemType Directory -Force -Path $Target | Out-Null
$Venv = Join-Path $Target "app-$($Info.version)"
if (Test-Path $Venv) { throw "$Venv already exists. Remove it first to reinstall the same version." }
& $Python -m venv $Venv
$VPy = Join-Path $Venv "Scripts\python.exe"
# Wheels collect in one local wheelhouse, so a small update bundle (application only) finds the runtime there.
$Wheels = Join-Path $Target "wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheels | Out-Null
Get-ChildItem (Join-Path $Bundle "wheelhouse") -Filter *.whl | ForEach-Object {
  $dest = Join-Path $Wheels $_.Name
  if (-not (Test-Path $dest)) { Copy-Item $_.FullName $dest }
}
& $VPy -m pip install --no-index --find-links $Wheels "trafficwb[probe,$($Info.ort)]==$($Info.version)"
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }

# 4. Model packs: add the bundle's versions; existing versions are kept.
$Models = Join-Path $Target "models"
$BundleModels = Join-Path $Bundle "models"
if (Test-Path $BundleModels) { Get-ChildItem $BundleModels -Directory | ForEach-Object {
  Get-ChildItem $_.FullName -Directory | ForEach-Object {
    $dest = Join-Path $Models (Join-Path $_.Parent.Name $_.Name)
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null; Copy-Item -Recurse $_.FullName $dest }
  }
} }

# 5. Check the install end to end with no network, before switching to it.
if (-not $SkipCheck) {
  $work = Join-Path $env:TEMP ("twb-check-" + [guid]::NewGuid())
  & $VPy (Join-Path $Bundle "offline_smoke.py") --work $work --models $Models
  if ($LASTEXITCODE -ne 0) { throw "The install check failed; the previous version stays in use." }
}

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
Write-Host "Installed. Start with $Target\start.cmd. The first account is created in the browser on this computer."
