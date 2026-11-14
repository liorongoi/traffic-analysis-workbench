; Setup wizard that adds the largest detector sizes to an installed LanePulse (Inno Setup 6).
; LanePulse-Setup holds at most 4.2 GB, so these come separately: YOLOX l and x, RF-DETR medium, D-FINE large and
; xlarge, RT-DETRv2 r34, r50 and r101, with their training starting weights. It finds LanePulse's install folder,
; runs install-detectors.ps1 from a temporary folder, and can be removed on its own. Nothing is downloaded.
;
; Built by .github/workflows/bundle.yml:
;   iscc /DDetDir=<folder with models\, training\, SHA256SUMS, install-detectors.ps1> /DAppVersion=<version> /DOutDir=<folder> packaging\detectors.iss

#ifndef DetDir
  #error Pass /DDetDir=<detectors folder>
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef OutDir
  #define OutDir "dist"
#endif
; LanePulse's own Setup (setup.iss): its uninstall entry says where it is installed.
#define LanePulseKey "Software\Microsoft\Windows\CurrentVersion\Uninstall\{6C1E3F0A-7A51-4E0B-9C8E-1F2B7D9A4E21}_is1"

[Setup]
AppId={{B4D2A7C9-3E61-4F58-8A0D-5C9E1B7F2A43}
AppName=LanePulse detectors (large sizes)
AppVersion={#AppVersion}
AppVerName=LanePulse detectors {#AppVersion}
DefaultDirName={code:LanePulseDir}
UsePreviousAppDir=no
DisableDirPage=yes
DisableProgramGroupPage=yes
CreateUninstallRegKey=yes
Uninstallable=yes
PrivilegesRequired=lowest
OutputDir={#OutDir}
OutputBaseFilename=LanePulse-Detectors-{#AppVersion}
; ONNX and PyTorch weights barely compress.
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=LanePulse detectors (large sizes)
SetupIconFile=lanepulse.ico
SetupLogging=yes

[Files]
Source: "{#DetDir}\*"; DestDir: "{tmp}\detectors"; Flags: recursesubdirs createallsubdirs ignoreversion
; Kept for the uninstall.
Source: "{#DetDir}\install-detectors.ps1"; DestDir: "{app}"; Flags: ignoreversion

[UninstallRun]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install-detectors.ps1"" -Target ""{app}"" -Uninstall"; Flags: runhidden waituntilterminated; RunOnceId: "RemoveDetectors"

[UninstallDelete]
Type: files; Name: "{app}\install-detectors.ps1"

[Code]
var
  Failed: Boolean;

function LanePulseDir(Param: String): String;
begin
  if RegQueryStringValue(HKCU, '{#LanePulseKey}', 'InstallLocation', Result) and (Result <> '') then
    Result := RemoveBackslashUnlessRoot(Result)
  else
    Result := ExpandConstant('{localappdata}\TrafficWorkbench');
end;

function InitializeSetup: Boolean;
begin
  Result := FileExists(LanePulseDir('') + '\current.txt');
  if not Result then
    SuppressibleMsgBox('LanePulse is not installed on this computer for this user.' + #13#10 + #13#10 +
      'Install LanePulse first (LanePulse-Setup), then run this again.', mbError, MB_OK, IDOK);
end;

function Quote(S: String): String;
begin
  StringChangeEx(S, '''', '''''', True);
  Result := '''' + S + '''';
end;

procedure RunInstaller;
var
  Log, Cmd, Problem: String;
  Reason: AnsiString;
  Code: Integer;
begin
  ForceDirectories(ExpandConstant('{app}\logs'));
  Log := ExpandConstant('{app}\logs\install-detectors.log');
  Problem := ExpandConstant('{app}\logs\install-detectors-problem.txt');
  DeleteFile(Problem);
  Cmd := '$env:PSModulePath=[Environment]::GetEnvironmentVariable(''PSModulePath'',''Machine''); ' +
    '$ErrorActionPreference=''Stop''; try { & ' + Quote(ExpandConstant('{tmp}\detectors\install-detectors.ps1')) +
    ' -Target ' + Quote(ExpandConstant('{app}')) + ' -From ' + Quote(ExpandConstant('{tmp}\detectors')) +
    ' *>&1 | Out-File -Encoding utf8 ' + Quote(Log) + '; exit 0 } catch { $_ | Out-File -Append -Encoding utf8 ' + Quote(Log) +
    '; [IO.File]::WriteAllText(' + Quote(Problem) + ', $_.Exception.Message); exit 1 }';
  WizardForm.StatusLabel.Caption := 'Checking and adding the detector sizes. This takes a minute or two...';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  try
    if not Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -Command "' + Cmd + '"', '',
                SW_HIDE, ewWaitUntilTerminated, Code) then
      Code := -1;
  finally
    WizardForm.ProgressGauge.Style := npbstNormal;
  end;
  if Code <> 0 then
  begin
    if LoadStringFromFile(Problem, Reason) and (Trim(String(Reason)) <> '') then
      Reason := #13#10 + #13#10 + 'Reason: ' + Reason
    else
      Reason := '';
    SuppressibleMsgBox('The detector sizes were not added (code ' + IntToStr(Code) + ').' + String(Reason) + #13#10 + #13#10 +
      'Details are in:' + #13#10 + Log + #13#10 + #13#10 + 'LanePulse itself is unchanged.', mbError, MB_OK, IDOK);
    Failed := True;
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  if Failed then Result := 1 else Result := 0;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then RunInstaller;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
  begin
    if Failed then
      WizardForm.FinishedLabel.Caption := 'The detector sizes were not added. Details are in ' +
        ExpandConstant('{app}\logs\install-detectors.log') + '.'
    else
      WizardForm.FinishedLabel.Caption := 'The large detector sizes are added. LanePulse offers them right away ' +
        '(no restart needed), and picks them on computers fast enough for them.';
  end;
end;
