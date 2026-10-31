# What can go wrong while installing LanePulse on Windows, tried with the setup wizard on a clean runner:
#   1. not enough disk space for a first install: Setup fails with a clear reason and leaves no icon that opens nothing;
#   2. a first install stopped part way (Setup closed or the computer switched off while it installs): the next Setup
#      finishes and removes what the stopped one left;
#   3. an update stopped part way while LanePulse runs: the running version keeps working, the next Setup finishes;
#   4. files cut short by an interrupted copy (a wheel, a model file, a missing SAM 3.1 file): Setup repairs them;
#   5. not enough disk space for an update (for the install, and for unpacking Setup itself): it fails with a clear
#      reason, and the installed version keeps working;
#   6. uninstall afterwards removes everything (except the data folder).
#   windows_failures.ps1 -Setup <LanePulse-Setup-x.exe> -Version <x>
param([Parameter(Mandatory)][string]$Setup, [Parameter(Mandatory)][string]$Version)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\uninstall_check.ps1"
$dir = "$env:LOCALAPPDATA\TrafficWorkbench"
$data = "$env:USERPROFILE\TrafficData"
$menu = Join-Path ([Environment]::GetFolderPath("Programs")) "LanePulse.lnk"
$wc = New-Object System.Net.WebClient; $wc.Proxy = $null
$fill = "C:\lanepulse-test-fill.bin"

function Step([string]$s) { Write-Host ""; Write-Host "===== $s" }
function Get-Health { try { return ($wc.DownloadString("http://127.0.0.1:8765/api/health") | ConvertFrom-Json) } catch { return $null } }
function Wait-Health([string]$what) {
  for ($i = 0; $i -lt 60; $i++) { $h = Get-Health; if ($h) { return $h }; Start-Sleep -Seconds 2 }
  Get-Content "$dir\logs\server.log" -ErrorAction SilentlyContinue | Select-Object -Last 30 | Out-Host
  throw "$($what): the server did not answer"
}
function Start-App([string]$what) {
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$dir\start.cmd`"" -WorkingDirectory $dir -WindowStyle Hidden
  return (Wait-Health $what)
}
function Stop-App {
  & "$dir\autostart.ps1" -Mode remove -Target $dir | Out-Host
  for ($i = 0; $i -lt 30 -and (Get-Health); $i++) { Start-Sleep -Seconds 1 }
  if (Get-Health) { throw "the server did not stop" }
}
function Current { return (Get-Content "$dir\current.txt" -Raw).Trim() }
function Show-Log([string]$what) {
  Get-Content "$dir\logs\install.log" -ErrorAction SilentlyContinue | Select-Object -Last 25 | Out-Host
  Get-Content "$env:RUNNER_TEMP\setup-$what.log" -ErrorAction SilentlyContinue | Select-Object -Last 25 | Out-Host
}
function Invoke-Setup([string]$what, [int]$minutes = 20) {
  $p = Start-Process -PassThru $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/LOG=$env:RUNNER_TEMP\setup-$what.log"
  if (-not $p.WaitForExit($minutes * 60 * 1000)) {
    Show-Log $what
    taskkill /T /F /PID $p.Id | Out-Host
    throw "$($what): Setup did not finish in $minutes minutes (waiting on a message box?)"
  }
  return $p.ExitCode
}
# Installed and working: every part loads, the server starts with this build, the Start menu entry is there.
function Assert-Installed([string]$what) {
  if (-not (Test-Path "$dir\current.txt")) { throw "$($what): not installed" }
  $venv = Current
  if (-not (Test-Path "$venv\Scripts\python.exe")) { throw "$($what): $venv is not a whole install" }
  if (-not (Test-Path $menu)) { throw "$($what): no Start menu entry" }
  & "$venv\Scripts\python.exe" "$dir\pc_check.py" --training | Select-Object -SkipLast 1 | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "$($what): pc_check failed" }
  $h = Start-App $what
  if ($h.version -ne $Version -or $h.install -ne (Split-Path -Leaf $venv)) { throw "$($what): the server is $($h.version) $($h.install), not $venv" }
  Stop-App
  $left = @(Get-ChildItem $dir -Filter "app-*.installing" -File)
  if ($left) { throw "$($what): a stopped install's folder is still there: $($left.Name -join ', ')" }
  Write-Host "$what OK: installed and working ($venv)."
}
# Starts Setup and stops it hard (as closing it from Task Manager, or a power cut, would) once the new Python
# environment is being filled.
function Stop-SetupMidway([string]$what) {
  $p = Start-Process -PassThru $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/LOG=$env:RUNNER_TEMP\setup-$what.log"
  $hit = $false
  for ($i = 0; $i -lt 1200; $i++) {
    $half = Get-ChildItem $dir -Filter "app-*.installing" -File -ErrorAction SilentlyContinue |
      Where-Object { Test-Path (Join-Path ($_.FullName -replace '\.installing$', '') "Lib\site-packages\numpy") }
    if ($half) { $hit = $true; break }
    if ($p.HasExited) { Show-Log $what; throw "$($what): Setup ended (code $($p.ExitCode)) before it could be stopped" }
    Start-Sleep -Milliseconds 500
  }
  if (-not $hit) { throw "$($what): the install never got to filling the Python environment" }
  Start-Sleep -Seconds 5
  taskkill /T /F /PID $p.Id | Out-Host
  # Anything the installer started that outlived its parent (pip, the install check).
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match "install\.ps1|pip install|offline_smoke|pc_check|is-[A-Z0-9]+\.tmp") } |
    ForEach-Object { Write-Host "stopping $($_.ProcessId) $($_.CommandLine)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 5
  Write-Host "$($what): Setup stopped while installing."
}
# Leaves about $gb GB free on C:.
function Set-FreeSpace([double]$gb) {
  Remove-Item $fill -Force -ErrorAction SilentlyContinue
  $free = (Get-PSDrive C).Free
  $size = [int64]($free - $gb * 1GB)
  if ($size -gt 0) { fsutil file createnew $fill $size | Out-Host }
  Write-Host ("C: now has {0:N1} GB free" -f ((Get-PSDrive C).Free / 1GB))
}
function Clear-FreeSpace { Remove-Item $fill -Force -ErrorAction SilentlyContinue }

New-Item -ItemType Directory -Force "$data\keep-me" | Out-Null

Step "1. First install with too little disk space"
Set-FreeSpace 5
$code = Invoke-Setup "lowdisk-first"
Clear-FreeSpace
Show-Log "lowdisk-first"
if ($code -eq 0) { throw "Setup reported success with too little disk space" }
$why = Get-Content "$dir\logs\install-problem.txt" -Raw -ErrorAction SilentlyContinue
Write-Host "Setup exit code $code; reason: $why"
if ($why -notmatch "free space") { throw "the reason does not say it is the disk space: '$why'" }
if (Test-Path $menu) { throw "a failed first install left a Start menu entry that would open nothing" }
if (Test-Path "$dir\current.txt") { throw "a failed first install was recorded as installed" }

Step "2. First install stopped part way, then Setup again"
Stop-SetupMidway "stopped-first"
if (Get-Health) { throw "a server runs after a stopped first install" }
if (Test-Path "$dir\current.txt") { throw "a stopped first install was recorded as installed" }
$code = Invoke-Setup "after-stopped-first"
if ($code -ne 0) { Show-Log "after-stopped-first"; throw "Setup after a stopped first install: exit code $code" }
Select-String -Path "$dir\logs\install.log" -Pattern "left by an install that did not finish" | Out-Host
Assert-Installed "after a stopped first install"
$apps = @(Get-ChildItem $dir -Directory -Filter "app-*")
if ($apps.Count -ne 1) { throw "expected one app folder, found: $($apps.Name -join ', ')" }

Step "3. Update stopped part way while LanePulse runs, then Setup again"
$good = Current
$h = Start-App "before the stopped update"
Stop-SetupMidway "stopped-update"
$h = Get-Health
if (-not $h -or $h.install -ne (Split-Path -Leaf $good)) { throw "the running version stopped working after a stopped update: $($h | ConvertTo-Json -Compress)" }
if ((Current) -ne $good) { throw "a stopped update switched current.txt to $(Current)" }
Stop-App
$h = Start-App "the installed version after a stopped update"
if ($h.install -ne (Split-Path -Leaf $good)) { throw "start.cmd no longer starts the installed version" }
$code = Invoke-Setup "after-stopped-update"
if ($code -ne 0) { Show-Log "after-stopped-update"; throw "Setup after a stopped update: exit code $code" }
Assert-Installed "after a stopped update"
$apps = @(Get-ChildItem $dir -Directory -Filter "app-*")
if ($apps.Count -ne 2) { throw "expected the new app folder and the previous one, found: $($apps.Name -join ', ')" }

Step "4. Files cut short by an interrupted copy, then Setup again"
$whl = Get-ChildItem "$dir\wheelhouse" -Filter "numpy-*.whl" | Select-Object -First 1
$fs = [IO.File]::Open($whl.FullName, "Open"); $fs.SetLength([int64]($fs.Length / 2)); $fs.Close()
$onnx = Get-ChildItem "$dir\models\yolox-s" -Recurse -Filter model.onnx | Select-Object -First 1
$fs = [IO.File]::Open($onnx.FullName, "Open"); $fs.SetLength(1000); $fs.Close()
$samFile = Get-ChildItem "$dir\models\sam3.1-w8a16" -Recurse -Filter "*.bin" | Select-Object -First 1
Remove-Item $samFile.FullName
$code = Invoke-Setup "repair-files"
if ($code -ne 0) { Show-Log "repair-files"; throw "Setup over damaged files: exit code $code" }
if ((Get-Item $whl.FullName).Length -le 1000 -or -not (Test-Path $samFile.FullName) -or (Get-Item $onnx.FullName).Length -le 1000) { throw "the damaged files were not repaired" }
Select-String -Path "$dir\logs\install.log" -Pattern "Repairing" | Out-Host
Assert-Installed "after repairing damaged files"

Step "5. Update with too little disk space (for the install, then for unpacking Setup itself)"
$good = Current
foreach ($case in @(@{ name = "lowdisk-update"; gb = 5 }, @{ name = "lowdisk-unpack"; gb = 1 })) {
  Set-FreeSpace $case.gb
  $code = Invoke-Setup $case.name 15
  Clear-FreeSpace
  Show-Log $case.name
  Write-Host "$($case.name): Setup exit code $code"
  if ($code -eq 0) { throw "$($case.name): Setup reported success with too little disk space" }
  if ((Current) -ne $good) { throw "$($case.name): current.txt changed to $(Current)" }
  $h = Start-App "the installed version after $($case.name)"
  if ($h.install -ne (Split-Path -Leaf $good)) { throw "$($case.name): the installed version no longer starts" }
  Stop-App
  if (-not (Test-Path $menu)) { throw "$($case.name): the Start menu entry of the installed version was removed" }
}
$code = Invoke-Setup "after-lowdisk"
if ($code -ne 0) { Show-Log "after-lowdisk"; throw "Setup after freeing space: exit code $code" }
Assert-Installed "after freeing disk space"

Step "6. Uninstall"
Test-Uninstall -Dir $dir -Data $data
Write-Host ""
Write-Host "All failure cases passed."
