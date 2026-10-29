# Offline install or update of LanePulse on a Windows workstation.
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
  [string]$AutoStart = "none",     # Start manually by default; boot is for an IT administrator
  [switch]$SkipCheck
)
$ErrorActionPreference = "Stop"
$Bundle = $PSScriptRoot
$Info = Get-Content (Join-Path $Bundle "bundle.json") -Raw | ConvertFrom-Json
Write-Host "LanePulse $($Info.version) for Python $($Info.python)"

# 1. Check every file against SHA256SUMS. .NET directly, not Get-FileHash, which is missing when this runs
#    under Windows PowerShell started from PowerShell 7 (its module path points at the wrong modules).
function Get-Sha256([string]$path) {
  $sha = [Security.Cryptography.SHA256]::Create()
  $fs = [IO.File]::OpenRead($path)
  try { return ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace "-", "").ToLower() } finally { $fs.Dispose(); $sha.Dispose() }
}
Get-Content (Join-Path $Bundle "SHA256SUMS") | ForEach-Object {
  $parts = $_ -split "  ", 2
  if ($parts.Count -eq 2) {
    $f = Join-Path $Bundle $parts[1]
    $h = Get-Sha256 $f
    if ($h -ne $parts[0]) { throw "Checksum mismatch: $($parts[1]). Copy the bundle again." }
  }
}
Write-Host "Bundle checksums verified."

# This computer: 64-bit Windows 10 or 11 (Setup itself refuses 32-bit Windows).
$os = [Environment]::OSVersion.Version
if ($os.Major -lt 10) { throw "LanePulse needs Windows 10 or 11; this computer runs Windows $os." }
Write-Host "Windows $os, 64-bit: $([Environment]::Is64BitOperatingSystem)"

# Room for the new version (its Python environment, with PyTorch when training is included) next to the previous one.
$needGB = if ($Info.app_only) { 1 } elseif ($Info.training) { 8 } else { 4 }
$root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Target))
try {
  $freeGB = [math]::Floor((New-Object IO.DriveInfo $root).AvailableFreeSpace / 1GB)
  if ($freeGB -lt $needGB) { throw "Not enough free space on $root for LanePulse: $freeGB GB free, about $needGB GB needed. Free some space and run Setup again." }
  Write-Host "Free space on ${root}: $freeGB GB (about $needGB GB needed)."
} catch [System.ArgumentException] { }

# Microsoft C++ runtime (msvcp140.dll and the rest). ONNX Runtime and PyTorch need a recent one, and the copy in
# C:\Windows\System32 is often older on computers where it was never updated (updating it needs an administrator).
# The bundle carries its own; it is kept in <install>\vc-runtime, so a small update finds it too, and put next to
# LanePulse's Python and next to the libraries that use it (ONNX Runtime, PyTorch, OpenVINO for SAM 3.1), where Windows
# looks before System32.
$KeptVc = Join-Path $Target "vc-runtime"
$BundleVc = Join-Path $Bundle "vc-runtime"
if (Test-Path $BundleVc) {
  New-Item -ItemType Directory -Force -Path $KeptVc | Out-Null
  Copy-Item (Join-Path $BundleVc "*.dll") $KeptVc -Force
}
function Get-FileVer([string]$p) {
  $i = [Diagnostics.FileVersionInfo]::GetVersionInfo($p)
  return [version]"$($i.FileMajorPart).$($i.FileMinorPart).$($i.FileBuildPart).$($i.FilePrivatePart)"
}
function Install-VcRuntime([string]$dir) {
  if (-not (Test-Path $KeptVc) -or -not (Test-Path $dir)) { return }
  foreach ($dll in Get-ChildItem $KeptVc -Filter *.dll) {
    $dest = Join-Path $dir $dll.Name
    if ((Test-Path $dest) -and ((Get-FileVer $dest) -ge (Get-FileVer $dll.FullName))) { continue }
    # A running LanePulse holds its copy open; it is the same version, and the restart below picks up any newer one.
    try { Copy-Item $dll.FullName $dest -Force }
    catch {
      # In use by a LanePulse server still running from this Python: a loaded DLL can be renamed, not overwritten.
      try {
        Rename-Item -LiteralPath $dest "$($dll.Name).old-$(Get-Date -Format yyyyMMddHHmmss)"
        Copy-Item $dll.FullName $dest -Force
      } catch { Write-Host "Could not replace $dest (in use); keeping the copy there." }
    }
  }
}

# 2. Python. A bundle with its own Python (python-runtime: a complete, unregistered copy) always uses that one,
#    copied into the install folder once per Python version, so no other Python on this computer is used or
#    changed, and an update never alters the Python the previous version runs on. Older bundles: the given Python,
#    the py launcher, or the installer shipped in the bundle.
# A Python found must really be there: after an uninstall Windows can still list one whose folder is gone.
$pyDir = Join-Path $Target "python$($Info.python -replace '\.','')"
function Test-Python([string]$exe) {
  if (-not $exe -or -not (Test-Path $exe)) { return $false }
  try { $v = (& $exe -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null).Trim() } catch { return $false }
  return $v -eq $Info.python
}
$Runtime = Join-Path $Bundle "python-runtime"
if (-not $Python -and $Info.python_runtime -and (Test-Path (Join-Path $Runtime "python.exe"))) {
  $own = Join-Path $Target "python-$($Info.python_runtime)"
  if (-not (Test-Path (Join-Path $own "python.exe"))) {
    Write-Host "Copying LanePulse's own Python $($Info.python_runtime) into $own..."
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    $part = "$own.partial"
    if (Test-Path $part) { Remove-Item -Recurse -Force $part }
    Copy-Item -Recurse $Runtime $part
    Rename-Item $part (Split-Path $own -Leaf)
  }
  $Python = Join-Path $own "python.exe"
  if (-not (Test-Python $Python)) {
    # E.g. blocked by a program policy on this computer: fall back to the Python installer below.
    Write-Host "LanePulse's own Python in $own does not start here; using the Python installer instead."
    $Python = ""
  }
}
if (-not $Python -and (Test-Python (Join-Path $pyDir "python.exe"))) { $Python = Join-Path $pyDir "python.exe" }
if (-not $Python) {
  try { $Python = (& py "-$($Info.python)" -c "import sys; print(sys.executable)" 2>$null).Trim() } catch { $Python = "" }
  if (-not (Test-Python $Python)) { $Python = "" }
}
if (-not $Python) {
  try { $Python = (& python -c "import sys; print(sys.executable)" 2>$null).Trim() } catch { $Python = "" }
  if (-not (Test-Python $Python)) { $Python = "" }
}
if ($Python) {
  try {
    $v = (& $Python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
    if ($v -ne $Info.python) { $Python = "" }
  } catch { $Python = "" }
}
if (-not $Python) {
  $existingPython = Join-Path (Join-Path $Target "python$($Info.python -replace '\.','')") "python.exe"
  if (Test-Path -LiteralPath $existingPython -PathType Leaf) {
    try {
      $v = (& $existingPython -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
      if ($v -eq $Info.python) { $Python = $existingPython }
    } catch { }
  }
}
if (-not $Python) {
  $installer = Get-ChildItem $Bundle -Filter "python-*.exe" | Sort-Object Name | Select-Object -First 1
  if (-not $installer) { throw "Python $($Info.python) is not installed and no python-*.exe is in the bundle." }
  # Quoted so a TargetDir with spaces stays one argument; Start-Process -Wait waits for the GUI installer.
  $pyArgs = @("/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_test=0", "Include_launcher=0", "Shortcuts=0", "TargetDir=`"$pyDir`"")
  Write-Host "Installing Python $($Info.python) for this user into $pyDir..."
  Start-Process -Wait -FilePath $installer.FullName -ArgumentList $pyArgs
  if (-not (Test-Python (Join-Path $pyDir "python.exe"))) {
    # Windows still thinks this Python is installed (its folder was deleted): remove that entry and try again.
    Write-Host "Python was registered but its files are missing; repairing the registration and installing again."
    Start-Process -Wait -FilePath $installer.FullName -ArgumentList "/quiet", "/uninstall"
    Start-Process -Wait -FilePath $installer.FullName -ArgumentList $pyArgs
  }
  $Python = Join-Path $pyDir "python.exe"
  if (-not (Test-Python $Python)) { throw "Python $($Info.python) could not be installed into $pyDir. Check the Python installer log and retry." }
  # Kept so the uninstaller can remove this Python cleanly.
  Copy-Item $installer.FullName (Join-Path $Target "python-installer.exe") -Force
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python was not found at '$Python'. Install Python $($Info.python) or repair the bundled runtime." }
Write-Host "Python: $Python"
# Only LanePulse's own Python folder is changed, never a Python that belongs to something else.
if ([IO.Path]::GetFullPath($Python).StartsWith([IO.Path]::GetFullPath($Target), [StringComparison]::OrdinalIgnoreCase)) {
  Install-VcRuntime (Split-Path $Python)
}

# 3. Wheels collect in one local wheelhouse, so a small update bundle (application only) finds the runtime there.
New-Item -ItemType Directory -Force -Path $Target | Out-Null
$Venv = Join-Path $Target "app-$($Info.version)"
# Keep the previous build intact for rollback. A same-version repair gets a fresh folder so reinstalling
# the same setup executable does not collide with the existing environment.
if (Test-Path $Venv) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $Venv = Join-Path $Target "app-$($Info.version)-repair-$stamp"
  if (Test-Path $Venv) { throw "A repair folder already exists at $Venv. Wait a moment and retry." }
}
$Wheels = Join-Path $Target "wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheels | Out-Null
Get-ChildItem (Join-Path $Bundle "wheelhouse") -Filter *.whl | ForEach-Object {
  $dest = Join-Path $Wheels $_.Name
  # The application wheel can be rebuilt under the same version number: always take the bundle's own.
  if (-not (Test-Path $dest) -or $_.Name -like "trafficwb-*") { Copy-Item $_.FullName $dest -Force }
}

# 4. Model packs: add the bundle's versions; existing versions are kept.
$Models = Join-Path $Target "models"
# Models trained on this computer that an uninstall set aside: back into the models folder.
$Kept = Join-Path $Target "trained-models"
if (Test-Path $Kept) {
  foreach ($pack in @(Get-ChildItem $Kept -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory })) {
    $dest = Join-Path $Models (Join-Path $pack.Parent.Name $pack.Name)
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null; Move-Item $pack.FullName $dest }
  }
  if (-not (Get-ChildItem $Kept -Recurse -File)) { Remove-Item -Recurse -Force $Kept }
  else { Write-Host "Some trained models in $Kept have the same name as installed ones and were left there." }
}
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
  # Messages a program writes to stderr (pip warnings, model loading notes) must not stop the install when the
  # output is redirected, as Windows PowerShell does then; failures are caught by the exit codes below.
  $ErrorActionPreference = "Continue"
  & $Python -m venv $Venv | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "Creating the Python environment failed." }
  $script:VPy = Join-Path $Venv "Scripts\python.exe"
  # Training on this computer (PyTorch) when the bundle carries it; --no-cache-dir so a rebuilt application wheel
  # of the same version is never taken from pip's cache.
  $extras = "probe,$build"
  if ($Info.training) { $extras += ",train" }
  & $script:VPy -m pip install --no-index --no-cache-dir --find-links $Wheels "trafficwb[$extras]==$($Info.version)" | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "pip install failed." }
  # Starting weights for training: kept in <install>\training (so a small update finds them too) and put next to
  # this environment's Python, where trafficwb.train looks.
  $kept = Join-Path $Target "training"
  $weights = Join-Path $Bundle "training\yolox_s.pth"
  if (Test-Path $weights) {
    New-Item -ItemType Directory -Force -Path $kept | Out-Null
    Copy-Item $weights $kept -Force
  }
  if (Test-Path (Join-Path $kept "yolox_s.pth")) {
    $share = Join-Path $Venv "share\lanepulse"
    New-Item -ItemType Directory -Force -Path $share | Out-Null
    Copy-Item (Join-Path $kept "yolox_s.pth") $share -Force
  }
  $site = Join-Path $Venv "Lib\site-packages"
  Install-VcRuntime (Join-Path $site "onnxruntime\capi")
  Install-VcRuntime (Join-Path $site "torch\lib")
  Install-VcRuntime (Join-Path $site "openvino\libs")
  if ($SkipCheck) { return "unchecked" }
  # First check that every compiled part loads on this computer, and say in plain words which one does not and why.
  $pcCheck = Join-Path $Bundle "pc_check.py"
  if (Test-Path $pcCheck) {
    $checkArgs = @($pcCheck)
    if ($Info.training) { $checkArgs += "--training" }
    $lines = & $script:VPy @checkArgs 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $lines | ForEach-Object { Write-Host $_ }
    if ($code -ne 0) {
      $why = ($lines | Where-Object { $_ -like "This computer could not load*" -or $_ -like "The Microsoft C++ runtime in use*" }) -join " "
      throw "$why The install check failed; the previous version stays in use."
    }
  }
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
    $ErrorActionPreference = "Continue"
    & (Join-Path $prevVenv "Scripts\trafficwb-admin.exe") backup --data $Data --to $BackupDir
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -ne 0) { throw "Backup before the update failed; nothing was switched." }
  }
}
$backupArgs = ""
if ($BackupDir) { $backupArgs = "--backup-dir `"$BackupDir`"" }
@"
@echo off
rem Starts LanePulse $($Info.version). Open http://127.0.0.1:$Port in a browser.
set TWB_MODELS=$Models
"$Venv\Scripts\trafficwb-web.exe" serve --data "$Data" --port $Port $backupArgs
"@ | Set-Content -Encoding ASCII (Join-Path $Target "start.cmd")
if (Test-Path (Join-Path $Target "current.txt")) { Copy-Item (Join-Path $Target "current.txt") (Join-Path $Target "previous.txt") -Force }
Set-Content -Path (Join-Path $Target "current.txt") -Value $Venv
Copy-Item (Join-Path $Bundle "autostart.ps1") (Join-Path $Target "autostart.ps1") -Force
if (Test-Path (Join-Path $Bundle "pc_check.py")) { Copy-Item (Join-Path $Bundle "pc_check.py") (Join-Path $Target "pc_check.py") -Force }
@"
@echo off
rem Measures how fast the models run on this computer (GPU and CPU). Takes a minute or two; changes nothing.
rem For a truer figure, add: --images "D:\a folder with about 20 typical images"
set TWB_MODELS=$Models
"$Venv\Scripts\trafficwb-speedcheck.exe" --out "$Target\logs\speed-check.txt" %*
pause
"@ | Set-Content -Encoding ASCII (Join-Path $Target "speed-check.cmd")
@"
@echo off
rem Adds a trained model (.zip from trafficwb-train) to this installation. Drag the .zip onto this file.
rem It does not change the model in use: evaluate it on the learning page and promote it there.
set PACK=%~1
if "%PACK%"=="" set /p PACK=Model .zip file: 
set TWB_MODELS=$Models
"$Venv\Scripts\trafficwb-models.exe" import "%PACK%"
pause
"@ | Set-Content -Encoding ASCII (Join-Path $Target "import-model.cmd")
$training = Join-Path $Bundle "TRAINING.txt"
if (Test-Path $training) { Copy-Item $training (Join-Path $Target "TRAINING.txt") -Force }
Write-Host "Installed $($Info.version)."

# 7. Existing automatic start. The task and shortcut keep their old names so existing installs are found.
# New installs start manually; an existing automatic start is preserved across updates.
$auto = Join-Path $Target "autostart.ps1"
# Windows can report no Startup folder (a profile without one, some managed computers).
$startupDir = [Environment]::GetFolderPath("Startup")
if (-not $startupDir -and $env:APPDATA) { $startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup" }
$hasLink = [bool]$startupDir -and (Test-Path (Join-Path $startupDir "Traffic Analysis Workbench.lnk"))
# Get-ScheduledTask can itself be missing or blocked on a locked-down computer: then there is no task to keep.
$hasTask = $false
try { $hasTask = [bool](Get-ScheduledTask -TaskName "Traffic Analysis Workbench" -ErrorAction SilentlyContinue) } catch { }
$wc = New-Object System.Net.WebClient; $wc.Proxy = $null
$running = $false
try { $null = $wc.DownloadString("http://127.0.0.1:$Port/api/health"); $running = $true } catch { }
if ($hasLink -or $hasTask) {
  & $auto -Mode restart -Target $Target -Port $Port
} elseif ($running) {
  # Started by hand or from the LanePulse icon: stop the old version and start this one in its place, so the icon
  # never opens the previous version after an update.
  Write-Host "Restarting the running server with $($Info.version)."
  & $auto -Mode restart -Target $Target -Port $Port
  Start-Process -FilePath (Join-Path $Target "start.cmd") -WorkingDirectory $Target -WindowStyle Minimized
} elseif ($AutoStart -ne "none") {
  & $auto -Mode $AutoStart -Target $Target -Port $Port
  if ($LASTEXITCODE -eq 2) { & $auto -Mode signin -Target $Target -Port $Port }
} else {
  Write-Host "Start with $Target\start.cmd."
}
Write-Host "Open http://127.0.0.1:$Port in a browser. No account or app sign-in is required."
