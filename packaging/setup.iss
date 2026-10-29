; Setup wizard for LanePulse (Inno Setup 6).
; Wraps the offline bundle: the wizard asks for the folders, unpacks the bundle to a temporary folder and runs
; install.ps1 from it, so the checks, the backup before an update and the automatic start are the same as a
; manual install. No administrator rights are needed. Nothing is downloaded.
;
; Built by .github/workflows/bundle.yml:
;   iscc /DBundleDir=<unpacked bundle folder> /DAppVersion=<version> /DOutDir=<folder> packaging\setup.iss

#ifndef BundleDir
  #error Pass /DBundleDir=<unpacked bundle folder>
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef OutDir
  #define OutDir "dist"
#endif
#define OpenArgs '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File ""{app}\autostart.ps1"" -Mode open'

[Setup]
AppId={{6C1E3F0A-7A51-4E0B-9C8E-1F2B7D9A4E21}
AppName=LanePulse
AppVersion={#AppVersion}
AppVerName=LanePulse {#AppVersion}
DefaultDirName={localappdata}\TrafficWorkbench
DefaultGroupName=LanePulse
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutDir}
OutputBaseFilename=LanePulse-Setup-{#AppVersion}
; The bundle is mostly compressed wheels already; fast compression keeps the build and the install quick.
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=LanePulse
SetupIconFile=lanepulse.ico
UninstallDisplayIcon={uninstallexe}
SetupLogging=yes

[Tasks]
Name: "desktopicon"; Description: "Put a LanePulse icon on the desktop"

[Files]
; Unpacked to the temporary folder, which Setup removes when it ends.
Source: "{#BundleDir}\*"; DestDir: "{tmp}\bundle"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "lanepulse.ico"; DestDir: "{app}"; Flags: ignoreversion

[InstallDelete]
; Start menu entries from before the app was called LanePulse.
Type: files; Name: "{autoprograms}\Lanemark - reset password.lnk"
Type: files; Name: "{autoprograms}\Lanemark.lnk"
Type: files; Name: "{autoprograms}\Lanemark - speed check.lnk"
Type: files; Name: "{autodesktop}\Lanemark.lnk"
Type: files; Name: "{autoprograms}\Traffic Analysis Workbench.url"
Type: files; Name: "{autoprograms}\Traffic Analysis Workbench - speed check.lnk"
; The first shortcut opened only the address and failed when the server was not running.
Type: files; Name: "{autoprograms}\Lanemark.url"

[Icons]
; "LanePulse" starts the server when it is not running, then opens it in the browser.
Name: "{autoprograms}\LanePulse"; Filename: "powershell.exe"; Parameters: "{#OpenArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\lanepulse.ico"; Comment: "Open LanePulse"; Flags: runminimized
Name: "{autodesktop}\LanePulse"; Filename: "powershell.exe"; Parameters: "{#OpenArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\lanepulse.ico"; Comment: "Open LanePulse"; Flags: runminimized; Tasks: desktopicon
Name: "{autoprograms}\LanePulse - speed check"; Filename: "{app}\speed-check.cmd"; WorkingDir: "{app}"

[Run]
Filename: "powershell.exe"; Parameters: "{#OpenArgs}"; Description: "Open LanePulse now"; Flags: postinstall nowait runhidden skipifsilent; Check: InstallOK

[UninstallRun]
; Its own RunOnceId: Windows keeps the uninstall entries of earlier versions, and an entry with an id already listed
; (the earlier "RemoveAutostart") would not run. It moves models trained here out of the models folder, which the
; earlier versions' lists delete, before any file is removed.
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\autostart.ps1"" -Mode uninstall -Target ""{app}"""; Flags: runhidden waituntilterminated; RunOnceId: "UninstallKeepTrained"
; The Python this installer put in the install folder, so Windows does not keep listing it after its files are gone.
Filename: "{app}\python-installer.exe"; Parameters: "/quiet /uninstall"; Flags: runhidden waituntilterminated skipifdoesntexist; RunOnceId: "RemovePython"

[UninstallDelete]
; Only the program. The data and backup folders, and models trained on this computer, are never removed.
Type: filesandordirs; Name: "{app}\app-*"
Type: filesandordirs; Name: "{app}\python*"
Type: filesandordirs; Name: "{app}\wheelhouse"
; Model packs: the uninstall step above removes the ones that came with LanePulse and moves models trained here to
; {app}\trained-models, which is kept.
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\startup"
Type: filesandordirs; Name: "{app}\training"
Type: filesandordirs; Name: "{app}\vc-runtime"
Type: filesandordirs; Name: "{app}\logs"
Type: files; Name: "{app}\*.cmd"
Type: files; Name: "{app}\current.txt"
Type: files; Name: "{app}\previous.txt"
Type: files; Name: "{app}\autostart.ps1"
Type: files; Name: "{app}\pc_check.py"
Type: files; Name: "{app}\TRAINING.txt"
Type: files; Name: "{app}\python-installer.exe"
Type: dirifempty; Name: "{app}"

[Code]
var
  FoldersPage: TInputDirWizardPage;
  Failed: Boolean;
  LogPath: String;

function InstallOK: Boolean;
begin
  Result := not Failed;
end;

procedure InitializeWizard;
begin
  FoldersPage := CreateInputDirPage(wpSelectDir, 'Data folders',
    'Where should LanePulse keep its data and backups?',
    'Missions, results and evidence go in the data folder. Pick a drive with plenty of space. ' +
    'The backup folder gets a copy of the data before every update and every night; another drive is best. ' +
    'Your source images and videos are only read, never moved.',
    False, '');
  FoldersPage.Add('Data folder:');
  FoldersPage.Add('Backup folder (leave empty for no backups):');
  FoldersPage.Values[0] := GetPreviousData('DataDir', ExpandConstant('{%USERPROFILE}\TrafficData'));
  FoldersPage.Values[1] := GetPreviousData('BackupDir', ExpandConstant('{%USERPROFILE}\TrafficBackups'));
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'DataDir', FoldersPage.Values[0]);
  SetPreviousData(PreviousDataKey, 'BackupDir', FoldersPage.Values[1]);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = FoldersPage.ID then
  begin
    if Trim(FoldersPage.Values[0]) = '' then
    begin
      SuppressibleMsgBox('Please choose a data folder.', mbError, MB_OK, IDOK);
      Result := False;
    end;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  Backup: String;
begin
  Backup := FoldersPage.Values[1];
  if Trim(Backup) = '' then Backup := '(no backups)';
  Result := MemoDirInfo + NewLine + NewLine +
    'Data folder:' + NewLine + Space + FoldersPage.Values[0] + NewLine + NewLine +
    'Backup folder:' + NewLine + Space + Backup + NewLine + NewLine + MemoTasksInfo + NewLine + NewLine +
    'Installing takes a few minutes. Nothing is downloaded.';
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
  Log := ExpandConstant('{app}\logs\install.log');
  LogPath := Log;
  // The reason, in plain words, when the install stops (what install.ps1 threw).
  Problem := ExpandConstant('{app}\logs\install-problem.txt');
  DeleteFile(Problem);
  // Windows PowerShell's own module path, in case Setup was started from PowerShell 7, whose path would load
  // the wrong modules. The log is written as UTF-8 so it reads anywhere.
  Cmd := '$env:PSModulePath=[Environment]::GetEnvironmentVariable(''PSModulePath'',''Machine''); ' +
    '$ErrorActionPreference=''Stop''; try { & ' + Quote(ExpandConstant('{tmp}\bundle\install.ps1')) +
    ' -Target ' + Quote(ExpandConstant('{app}')) + ' -Data ' + Quote(FoldersPage.Values[0]) +
    ' -BackupDir ' + Quote(FoldersPage.Values[1]) + ' -AutoStart none' +
    ' *>&1 | Out-File -Encoding utf8 ' + Quote(Log) + '; exit 0 } catch { $_ | Out-File -Append -Encoding utf8 ' + Quote(Log) +
    '; [IO.File]::WriteAllText(' + Quote(Problem) + ', $_.Exception.Message); exit 1 }';
  WizardForm.StatusLabel.Caption := 'Checking this computer and the files, setting up LanePulse (it brings its own Python) and testing the install. This takes a few minutes...';
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
    SuppressibleMsgBox('The installation did not finish (code ' + IntToStr(Code) + ').' + String(Reason) + #13#10 + #13#10 +
           'Details are in:' + #13#10 + Log + #13#10 + #13#10 +
           'Your data and previous version have been kept. Close LanePulse and retry the installer. If it fails again, share install.log.', mbError, MB_OK, IDOK);
    Failed := True;
  end;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
  begin
    if Failed then
      WizardForm.FinishedLabel.Caption := 'The installation did not finish. Details are in ' + LogPath + '.'
    else
      WizardForm.FinishedLabel.Caption := 'LanePulse is installed.' + #13#10 + #13#10 +
        'To open it, use the LanePulse icon on the desktop or in the Start menu. It opens in your browser at ' +
        'http://127.0.0.1:8765.';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then RunInstaller;
end;
