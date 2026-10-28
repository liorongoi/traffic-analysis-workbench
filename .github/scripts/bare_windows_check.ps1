# Runs inside a bare Windows Server Core container: no Visual C++ runtime, no Python, no GPU, no extras.
# Installs LanePulse with the setup wizard exactly as on a new PC, then checks that every part loads, that the
# install check (scan, detect, export, backup) passed, and that the server starts and answers.
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "uninstall_check.ps1")
$sys = Join-Path $env:SystemRoot "System32"
Write-Host "Windows $([Environment]::OSVersion.Version), user $env:USERNAME"
foreach ($d in "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "d3d12.dll") {
  $p = Join-Path $sys $d
  if (Test-Path $p) { Write-Host "  System32\$d $((Get-Item $p).VersionInfo.FileVersion)" } else { Write-Host "  System32\${d}: not present" }
}
if (Get-Command python -ErrorAction SilentlyContinue) { Write-Host "  A Python is on PATH" } else { Write-Host "  No Python on this computer" }

$exe = (Get-ChildItem C:\setup -Filter *.exe | Select-Object -First 1).FullName
$dir = Join-Path $env:LOCALAPPDATA "TrafficWorkbench"
$p = Start-Process -PassThru $exe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/LOG=C:\setup.log"
if (-not $p.WaitForExit(40 * 60 * 1000)) { throw "Setup did not finish in 40 minutes" }
Get-Content (Join-Path $dir "logs\install.log") -ErrorAction SilentlyContinue | Out-Host
if (Test-Path (Join-Path $dir "logs\install-problem.txt")) {
  throw "Setup stopped on bare Windows: $(Get-Content (Join-Path $dir 'logs\install-problem.txt') -Raw)"
}
if ($p.ExitCode -ne 0 -or -not (Test-Path (Join-Path $dir "current.txt"))) {
  Get-Content C:\setup.log -Tail 40 | Out-Host
  throw "Setup did not install LanePulse on bare Windows (exit code $($p.ExitCode))"
}
$venv = (Get-Content (Join-Path $dir "current.txt") -Raw).Trim()

# Every compiled part, and where the C++ runtime came from (System32 has none here).
$chk = & "$venv\Scripts\python.exe" (Join-Path $dir "pc_check.py") --training
$code = $LASTEXITCODE
$chk | Out-Host
if ($code -ne 0) { throw "pc_check failed on bare Windows" }

# Training is ready (PyTorch and the starting weights).
& "$venv\Scripts\python.exe" -c "from trafficwb import training_env as t; s = t.status(wait=True); print(s); assert s['state'] == 'ready' and s['coco_weights'], s"
if ($LASTEXITCODE -ne 0) { throw "training is not ready on bare Windows" }

# The server, started the way start.cmd starts it, answers and serves the screens.
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$dir\start.cmd`"" -WorkingDirectory $dir -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 90 -and -not $ok; $i++) {
  try { $h = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8765/api/health"; $ok = $true } catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) { Get-Content (Join-Path $dir "logs\server.log") -Tail 40 -ErrorAction SilentlyContinue | Out-Host; throw "the server did not answer on bare Windows" }
Write-Host "Health: $($h.Content)"
$page = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8765/"
if ($page.StatusCode -ne 200 -or $page.Content -notmatch "<div id=") { throw "the screens did not load on bare Windows" }
Write-Host "Bare Windows OK: installed, every part loads, install check passed, server answering."

# Uninstall while the server runs and starts at sign-in, with the data folder in use and a model trained here.
$data = Join-Path $env:USERPROFILE "TrafficData"
New-Item -ItemType Directory -Force (Join-Path $data "keep-me") | Out-Null
& (Join-Path $dir "autostart.ps1") -Mode remove -Target $dir | Out-Host
& (Join-Path $dir "autostart.ps1") -Mode signin -Target $dir | Out-Host   # starts it again through the restart loop
$model = New-TrainedModel $dir
Test-Uninstall -Dir $dir -Data $data -KeepModel $model

# Install again over what was kept, check it, and uninstall once more.
Remove-Item -Recurse -Force (Join-Path $dir "models")
$p = Start-Process -PassThru $exe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/LOG=C:\setup2.log"
if (-not $p.WaitForExit(40 * 60 * 1000)) { throw "The second install did not finish in 40 minutes" }
if (Test-Path (Join-Path $dir "logs\install-problem.txt")) { throw "The second install stopped: $(Get-Content (Join-Path $dir 'logs\install-problem.txt') -Raw)" }
if (-not (Test-Path (Join-Path $dir "current.txt"))) { Get-Content C:\setup2.log -Tail 40 | Out-Host; throw "The second install did not install LanePulse" }
$venv = (Get-Content (Join-Path $dir "current.txt") -Raw).Trim()
& "$venv\Scripts\python.exe" (Join-Path $dir "pc_check.py") | Select-Object -Last 1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pc_check failed after installing again" }
Write-Host "Install after uninstall OK."
Test-Uninstall -Dir $dir -Data $data
Write-Host "Bare Windows OK: install, uninstall, install again, uninstall."
