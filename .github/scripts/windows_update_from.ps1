# Updating from an older LanePulse build: installs that build with its own Setup, starts it, adds data and a model
# trained here, then runs this build's Setup over it while it runs, and checks the update (this version running, data
# and trained model kept, the older version kept for rollback), then uninstalls.
#   windows_update_from.ps1 -Old <older Setup.exe> -New <this Setup.exe> -Version <x>
param([Parameter(Mandatory)][string]$Old, [Parameter(Mandatory)][string]$New, [Parameter(Mandatory)][string]$Version)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\uninstall_check.ps1"
$dir = "$env:LOCALAPPDATA\TrafficWorkbench"
$data = "$env:USERPROFILE\TrafficData"
$menu = Join-Path ([Environment]::GetFolderPath("Programs")) "LanePulse.lnk"
$wc = New-Object System.Net.WebClient; $wc.Proxy = $null
function Get-Health {
  try { return ($wc.DownloadString("http://127.0.0.1:8765/api/health") | ConvertFrom-Json) } catch {
    # The first builds had no /api/health (404): a server is running, of an unknown older build.
    $e = $_.Exception
    while ($e -and -not ($e -is [System.Net.WebException])) { $e = $e.InnerException }
    if ($e -and $e.Response) { return [pscustomobject]@{ version = "older (no /api/health)"; install = "" } }
    return $null
  }
}
function Wait-Health {
  for ($i = 0; $i -lt 60; $i++) { $h = Get-Health; if ($h) { return $h }; Start-Sleep -Seconds 2 }
  Get-Content "$dir\logs\server.log" -ErrorAction SilentlyContinue | Select-Object -Last 30 | Out-Host
  return $null
}
function Invoke-Setup([string]$exe, [string]$what) {
  $p = Start-Process -PassThru $exe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/LOG=$env:RUNNER_TEMP\setup-$what.log"
  $done = $p.WaitForExit(25 * 60 * 1000)
  Get-Content "$dir\logs\install.log" -ErrorAction SilentlyContinue | Select-Object -Last 15 | Out-Host
  if (-not $done) { Get-Content "$env:RUNNER_TEMP\setup-$what.log" | Select-Object -Last 30 | Out-Host; throw "$($what): Setup did not finish" }
  return $p.ExitCode
}

Write-Host "===== The older build"
$code = Invoke-Setup $Old "older"
if ($code -ne 0 -or -not (Test-Path "$dir\current.txt")) { Get-Content "$env:RUNNER_TEMP\setup-older.log" | Select-Object -Last 30; throw "the older build did not install (exit code $code)" }
$oldVenv = (Get-Content "$dir\current.txt" -Raw).Trim()
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$dir\start.cmd`"" -WorkingDirectory $dir -WindowStyle Hidden
$h = Wait-Health
if (-not $h) { throw "the older build does not start" }
Write-Host "Older build running: $($h | ConvertTo-Json -Compress) from $oldVenv"
New-Item -ItemType Directory -Force "$data\keep-me" | Out-Null
$model = New-TrainedModel $dir

Write-Host "===== This build over it, while it runs"
$code = Invoke-Setup $New "update"
if ($code -ne 0) {
  Write-Host "----- install.log"
  Get-Content "$dir\logs\install.log" -ErrorAction SilentlyContinue | Select-Object -Last 250 | Out-Host
  throw "the update failed (exit code $code)"
}
# The install check is tried twice when it stops; show why the first try stopped.
Select-String -Path "$dir\logs\install.log" -Pattern "The install check stopped" -Context 40, 1 | Out-Host
$venv = (Get-Content "$dir\current.txt" -Raw).Trim()
if ($venv -eq $oldVenv) { throw "the update did not switch to the new version" }
if ((Get-Content "$dir\previous.txt" -Raw).Trim() -ne $oldVenv) { throw "the older version is not kept for rollback" }
if (-not (Test-Path $menu)) { throw "no Start menu entry after the update" }
if (-not (Test-Path "$data\keep-me")) { throw "the update lost the data folder" }
if (-not (Test-Path "$dir\models\vehicle-trained-test\1\model.json")) { throw "the update lost the model trained here" }
& "$venv\Scripts\python.exe" "$dir\pc_check.py" --training | Select-Object -SkipLast 1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw "pc_check failed after the update" }
# The running older server is replaced by the new one (or, if it was not replaced, the icon replaces it).
$h = Wait-Health
if (-not $h -or $h.install -ne (Split-Path -Leaf $venv)) {
  Write-Host "Running after the update: $($h | ConvertTo-Json -Compress); opening as the icon does."
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$dir\autostart.ps1" -Mode open -Target $dir 2>&1 | Out-Host
  $h = Wait-Health
}
if (-not $h -or $h.version -ne $Version -or $h.install -ne (Split-Path -Leaf $venv)) { throw "the new version is not the one running: $($h | ConvertTo-Json -Compress)" }
Write-Host "Update OK: $($h | ConvertTo-Json -Compress)"

Write-Host "===== Uninstall"
Test-Uninstall -Dir $dir -Data $data -KeepModel $model
