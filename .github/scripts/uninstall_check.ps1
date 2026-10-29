# Dot-sourced by the Windows installer tests. Test-Uninstall runs LanePulse's uninstaller silently and checks that it
# removed everything it should and kept what it must: the data folder and models trained on this computer.
function Test-Uninstall([string]$Dir, [string]$Data, [string]$KeepModel = "") {
  $unins = Join-Path $Dir "unins000.exe"
  if (-not (Test-Path $unins)) { throw "No uninstaller at $unins" }
  # The uninstaller hands over to a copy of itself in TEMP and exits; it is done when unins000.exe is gone.
  $u = Start-Process -PassThru $unins -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES"
  $null = $u.WaitForExit(5 * 60 * 1000)
  for ($i = 0; $i -lt 120 -and (Test-Path $unins); $i++) { Start-Sleep -Seconds 2 }
  if (Test-Path $unins) { throw "The uninstaller did not finish" }
  Start-Sleep -Seconds 10   # let stopped processes finish exiting
  $problems = @()

  # Program files: nothing left, except a model trained here when there is one.
  if (Test-Path $Dir) {
    $left = @(Get-ChildItem $Dir -Recurse -Force | ForEach-Object { $_.FullName.Substring($Dir.Length + 1) })
    $allowed = @()
    if ($KeepModel) { $allowed = @(($KeepModel -split "\\")[0], (Split-Path $KeepModel -Parent), $KeepModel) + @(Get-ChildItem (Join-Path $Dir $KeepModel) -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName.Substring($Dir.Length + 1) }) }
    $extra = @($left | Where-Object { $allowed -notcontains $_ })
    if ($extra) { $problems += "files left in ${Dir}: $($extra -join ', ')" }
  }
  if ($KeepModel -and -not (Test-Path (Join-Path (Join-Path $Dir $KeepModel) "model.json"))) { $problems += "the model trained on this computer was deleted" }

  # Nothing still running from the install folder, and the server is stopped.
  $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Dir + "\", [StringComparison]::OrdinalIgnoreCase) })
  if ($running) { $problems += "still running: $(($running | ForEach-Object { "$($_.ProcessName) $($_.Id)" }) -join ', ')" }
  # Also a command window still running the restart loop (cmd.exe itself lives in System32).
  $loops = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and $_.CommandLine -like "*$Dir*" -and $_.ProcessId -ne $PID })
  if ($loops) { $problems += "still running: $(($loops | ForEach-Object { $_.CommandLine }) -join ' | ')" }
  $wc = New-Object System.Net.WebClient; $wc.Proxy = $null
  try { $null = $wc.DownloadString("http://127.0.0.1:8765/api/health"); $problems += "the server still answers on port 8765" } catch { }

  # Shortcuts, automatic start and Windows' list of installed programs.
  $links = @()
  foreach ($f in "Programs", "Desktop", "Startup") {
    $p = [Environment]::GetFolderPath($f)
    if ($p) { $links += Get-ChildItem $p -Filter "*.lnk" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "LanePulse|Lanemark|Traffic Analysis Workbench" } }
  }
  if ($env:APPDATA) { $links += Get-ChildItem (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup") -Filter "*.lnk" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "Traffic Analysis Workbench" } }
  if ($links) { $problems += "shortcuts left: $(($links | ForEach-Object FullName | Sort-Object -Unique) -join ', ')" }
  try { if (Get-ScheduledTask -TaskName "Traffic Analysis Workbench" -ErrorAction SilentlyContinue) { $problems += "the power-on task is still there" } } catch { }
  $reg = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{6C1E3F0A-7A51-4E0B-9C8E-1F2B7D9A4E21}_is1"
  if (Test-Path $reg) { $problems += "still listed in Windows' installed programs" }

  # Kept on purpose: the data folder and its contents.
  if (-not (Test-Path (Join-Path $Data "keep-me"))) { $problems += "the data folder or its contents were removed" }

  if ($problems) { throw "Uninstall problems: $($problems -join '; ')" }
  Write-Host "Uninstall OK: program, shortcuts, automatic start, Windows' program list and running server all gone; data folder$(if ($KeepModel) { ' and trained model' }) kept."
}

# A stand-in for a model trained on this computer: its card records the training.
# Returns where the uninstaller is expected to keep it.
function New-TrainedModel([string]$Dir) {
  $d = Join-Path $Dir "models\vehicle-trained-test\1"
  New-Item -ItemType Directory -Force -Path $d | Out-Null
  '{"name": "vehicle-trained-test", "version": "1", "training": {"dataset": {"name": "test"}}}' | Set-Content (Join-Path $d "model.json")
  return "trained-models\vehicle-trained-test\1"
}
