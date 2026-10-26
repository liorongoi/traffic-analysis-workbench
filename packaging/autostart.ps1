# Start Lanemark automatically (ENV-08). Run from the install folder, in PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File autostart.ps1 -Mode status
#   powershell -ExecutionPolicy Bypass -File autostart.ps1 -Mode signin          # no administrator rights needed
#   powershell -ExecutionPolicy Bypass -File autostart.ps1 -Mode boot            # administrator: starts at power-on
#   powershell -ExecutionPolicy Bypass -File autostart.ps1 -Mode restart         # e.g. after an update
#   powershell -ExecutionPolicy Bypass -File autostart.ps1 -Mode remove
#
# signin:  the normal choice. Starts when you sign in to Windows, minimised, and restarts the server 10 s after
#          it stops. Needs no administrator rights. It also starts the server now.
# restart: stops the running server; it starts again by itself 10 s later, with the version start.cmd names.
# remove:  removes the automatic start and stops the server.
# boot:    for an IT administrator: a scheduled task that starts at power-on, before anyone signs in.
#          Creating it needs administrator rights. It runs as a Windows account you name (asked for its password),
#         so source folders on network shares stay reachable. -System runs it as the local SYSTEM account instead,
#         which cannot read network shares.
# Both write the server's output to <Target>\logs\server.log. Nothing is downloaded.
param(
  [ValidateSet("status", "signin", "boot", "restart", "remove")]
  [string]$Mode = "status",
  [string]$Target = $PSScriptRoot,
  [string]$User = "",
  [switch]$System,
  [int]$Port = 0
)
$ErrorActionPreference = "Stop"
# The task and shortcut keep the names from before the app was called Lanemark, so existing installs are found.
$TaskName = "Traffic Analysis Workbench"
$Start = Join-Path $Target "start.cmd"
$Runner = Join-Path $Target "run-server.cmd"
$Logs = Join-Path $Target "logs"
$Link = Join-Path ([Environment]::GetFolderPath("Startup")) "Traffic Analysis Workbench.lnk"

function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  return ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Port {
  if ($Port -gt 0) { return $Port }
  if (Test-Path $Start) {
    $m = Select-String -Path $Start -Pattern "--port (\d+)" | Select-Object -First 1
    if ($m) { return [int]$m.Matches[0].Groups[1].Value }
  }
  return 8765
}

function Test-Server([int]$p) {
  # Straight to this computer: never through a proxy.
  try {
    $wc = New-Object System.Net.WebClient
    $wc.Proxy = $null
    $null = $wc.DownloadString("http://127.0.0.1:$p/api/health")
    return $true
  } catch { return $false }
}

function Write-Runner {
  # Runs start.cmd with its output in the log, and starts it again 10 s after it stops.
  New-Item -ItemType Directory -Force -Path $Logs | Out-Null
  $log = Join-Path $Logs "server.log"
  $body = @"
@echo off
rem Written by autostart.ps1. Starts the server and restarts it if it stops.
:again
for %%F in ("$log") do if %%~zF GTR 10485760 move /y "$log" "$log.old" > nul
echo [%date% %time%] starting >> "$log"
call "$Start" >> "$log" 2>&1
echo [%date% %time%] stopped with code %errorlevel%, restarting in 10 s >> "$log"
timeout /t 10 /nobreak > nul
goto again
"@
  $body | Set-Content -Encoding ASCII $Runner
}

function Show-Status {
  $p = Get-Port
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  Write-Host "Install folder:        $Target"
  Write-Host "Administrator rights:  $(if (Test-Admin) { 'yes (boot mode can be set up)' } else { 'no (sign-in mode only)' })"
  Write-Host "Starts at sign-in:     $(if (Test-Path $Link) { 'yes' } else { 'no' })"
  if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "Starts at power-on:    yes, as $($task.Principal.UserId), state $($task.State), last result $($info.LastTaskResult)"
  } else {
    Write-Host "Starts at power-on:    no"
  }
  Write-Host "Server answering:      $(if (Test-Server $p) { "yes, http://127.0.0.1:$p" } else { "no (port $p)" })"
  if (Test-Path (Join-Path $Logs "server.log")) { Write-Host "Log:                   $(Join-Path $Logs 'server.log')" }
}

function Stop-Runner {
  # The window that restarts the server; stopped first so it does not start the server again.
  Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*run-server.cmd*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Remove-All {
  if (Test-Path $Link) {
    Remove-Item $Link -Force
    Write-Host "Removed the sign-in start."
    Stop-Runner
    Stop-Server
  }
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if ($task) {
    if (-not (Test-Admin)) { throw "Removing the power-on task needs administrator rights. Open PowerShell as administrator and run this again." }
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed the power-on task."
  }
  # Also stop a server that the installer launched directly, without a startup link or task.
  Stop-Runner
  Stop-Server
}

function Stop-Server {
  # Stopping the task ends its window but can leave the server itself running; stop whatever serves our port.
  # Processing resumes where it left off at the next start.
  $p = Get-Port
  # One process can listen on both IPv4 and IPv6: stop each process once.
  $ids = @(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
  foreach ($id in $ids) {
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    if ($proc -and ($proc.ProcessName -like "*trafficwb*" -or $proc.ProcessName -like "python*")) {
      Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
      Write-Host "Stopped the server (process $($proc.Id))."
    }
  }
}

if ($Mode -eq "status") { Show-Status; exit 0 }
if ($Mode -eq "remove") { Remove-All; exit 0 }
if ($Mode -eq "restart") {
  Stop-Server
  Write-Host "The server starts again within about 10 s if automatic start is set up; otherwise run $Start."
  exit 0
}
if (-not (Test-Path $Start)) { throw "$Start not found. Run install.ps1 first, or pass -Target with the install folder." }

if ($Mode -eq "signin") {
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "The power-on task is already set up; the server must not be started twice. Run -Mode remove first."
  }
  Write-Runner
  $sh = New-Object -ComObject WScript.Shell
  $lnk = $sh.CreateShortcut($Link)
  $lnk.TargetPath = $Runner
  $lnk.WorkingDirectory = $Target
  $lnk.WindowStyle = 7                      # minimised
  $lnk.Description = "Lanemark server"
  $lnk.Save()
  Write-Host "The server will start when you sign in to Windows (minimised window, log in $Logs)."
  $p = Get-Port
  if (Test-Server $p) {
    Write-Host "A server is already answering on port $p; it was left as it is."
  } else {
    Start-Process -FilePath $Link -WindowStyle Minimized
    for ($i = 0; $i -lt 60 -and -not (Test-Server $p); $i++) { Start-Sleep -Seconds 2 }
    if (Test-Server $p) { Write-Host "Server started: open http://127.0.0.1:$p" } else { Write-Host "The server has not answered yet; see $Logs\server.log" }
  }
  exit 0
}

# boot
if (-not (Test-Admin)) {
  Write-Host "Starting at power-on needs administrator rights, which this account does not have."
  Write-Host "Either open PowerShell as administrator and run this again, or use -Mode signin (no rights needed)."
  exit 2
}
if (Test-Path $Link) { Remove-Item $Link -Force; Write-Host "Removed the sign-in start (power-on replaces it)." }
Write-Runner
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$Runner`"" -WorkingDirectory $Target
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -MultipleInstances IgnoreNew
if ($System) {
  Write-Host "Running as SYSTEM: source folders on network shares will not be readable."
  $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Limited
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
} else {
  if (-not $User) { $User = "$env:USERDOMAIN\$env:USERNAME" }
  $cred = Get-Credential -UserName $User -Message "Windows account the server runs as (it needs read access to the source folders)"
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited -Force | Out-Null
}
Write-Host "The server will start at power-on and restart 10 s after it stops (log in $Logs)."
Write-Host "Starting it now..."
Start-ScheduledTask -TaskName $TaskName
$p = Get-Port
for ($i = 0; $i -lt 60 -and -not (Test-Server $p); $i++) { Start-Sleep -Seconds 2 }
if (Test-Server $p) { Write-Host "Server answering on http://127.0.0.1:$p" } else { Write-Host "The server has not answered yet; see $Logs\server.log" }
