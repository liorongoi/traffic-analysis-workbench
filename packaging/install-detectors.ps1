# Adds the largest detector sizes to an installed LanePulse (or removes them again). Run by the detectors wizard
# (detectors.iss); can also be run by hand from the unpacked folder:
#   powershell -ExecutionPolicy Bypass -File install-detectors.ps1 -Target C:\TrafficWorkbench
#   powershell -ExecutionPolicy Bypass -File install-detectors.ps1 -Target C:\TrafficWorkbench -Uninstall
# What it carries: models\<pack>\<version> (checked against SHA256SUMS) and training\*.pth (starting weights).
# The packs go in <Target>\models, where LanePulse finds them at once (no restart); the weights go in <Target>\training,
# which every LanePulse update copies into its new version, and into the running version now. Nothing is downloaded.
param(
  [Parameter(Mandatory)][string]$Target,
  [string]$From = $PSScriptRoot,
  [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$Models = Join-Path $Target "models"
$Kept = Join-Path $Target "training"
$List = Join-Path $Target "detectors-installed.json"

function Get-Venv {
  $cur = Join-Path $Target "current.txt"
  if (-not (Test-Path $cur)) { return $null }
  $v = (Get-Content $cur -Raw).Trim()
  if ($v -and (Test-Path $v)) { return $v } else { return $null }
}

if ($Uninstall) {
  if (-not (Test-Path $List)) { Write-Host "Nothing to remove."; exit 0 }
  $l = Get-Content $List -Raw | ConvertFrom-Json
  foreach ($p in @($l.packs)) {
    $d = Join-Path $Models $p
    if (Test-Path $d) { Remove-Item -Recurse -Force $d; Write-Host "Removed $d" }
    $parent = Split-Path $d
    if ((Test-Path $parent) -and -not (Get-ChildItem $parent -Force)) { Remove-Item $parent }
  }
  $venv = Get-Venv
  foreach ($w in @($l.weights)) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Kept $w)
    if ($venv) { Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $venv "share\lanepulse\$w") }
  }
  Remove-Item -Force $List
  exit 0
}

$venv = Get-Venv
if (-not $venv) { throw "LanePulse is not installed in $Target. Install LanePulse first (LanePulse-Setup), then run this again." }

# Every file whole, before anything is copied.
$sha = [Security.Cryptography.SHA256]::Create()
try {
  Get-Content (Join-Path $From "SHA256SUMS") | ForEach-Object {
    $parts = $_ -split "  ", 2
    if ($parts.Count -eq 2) {
      $fs = [IO.File]::OpenRead((Join-Path $From $parts[1]))
      try { $h = ([BitConverter]::ToString($sha.ComputeHash($fs)) -replace "-", "").ToLower() } finally { $fs.Dispose() }
      if ($h -ne $parts[0]) { throw "Checksum mismatch: $($parts[1]). Copy the detectors installer again." }
    }
  }
} finally { $sha.Dispose() }
Write-Host "Checksums verified."

# Room for them.
$need = (Get-ChildItem $From -Recurse -File | Measure-Object Length -Sum).Sum
$root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Target))
try {
  $free = (New-Object IO.DriveInfo $root).AvailableFreeSpace
  if ($free -lt $need + 512MB) {
    throw ("Not enough free space on {0}: {1:N1} GB free, about {2:N1} GB needed. Free some space and run it again." -f $root, ($free / 1GB), (($need + 512MB) / 1GB))
  }
} catch [System.ArgumentException] { }

$packs = @()
foreach ($pack in @(Get-ChildItem (Join-Path $From "models") -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory })) {
  $rel = Join-Path $pack.Parent.Name $pack.Name
  $dest = Join-Path $Models $rel
  $part = "$dest.partial"
  if (Test-Path $part) { Remove-Item -Recurse -Force $part }
  if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }      # an earlier copy, possibly cut short: replaced
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  Copy-Item -Recurse $pack.FullName $part
  Rename-Item $part (Split-Path $dest -Leaf)
  Write-Host "Installed $rel"
  $packs += ($rel -replace "\\", "/")
}
$weights = @()
New-Item -ItemType Directory -Force -Path $Kept | Out-Null
$share = Join-Path $venv "share\lanepulse"
New-Item -ItemType Directory -Force -Path $share | Out-Null
foreach ($w in @(Get-ChildItem (Join-Path $From "training") -Filter *.pth -ErrorAction SilentlyContinue)) {
  Copy-Item $w.FullName $Kept -Force
  Copy-Item $w.FullName $share -Force
  $weights += $w.Name
}
@{ packs = $packs; weights = $weights } | ConvertTo-Json | Set-Content -Encoding utf8 $List

# LanePulse sees them: its own model list.
$cli = Join-Path $venv "Scripts\trafficwb-models.exe"
if (Test-Path $cli) {
  $listed = (& $cli --models $Models list) -join "`n"
  Write-Host $listed
  foreach ($p in $packs) {
    $name = ($p -split "/")[0]
    if ($listed -notmatch [regex]::Escape($name)) { throw "LanePulse does not list $name after installing it." }
  }
}
Write-Host "Added $($packs.Count) detector sizes and $($weights.Count) training starting weights."
