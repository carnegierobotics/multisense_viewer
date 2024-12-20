; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "MultiSense Viewer"
#define MyAppVersion "1"
#define MyAppPublisher "Carnegie Robotics"
#define MyAppURL "https://www.carnegierobotics.com"
#define MyAppExeName "MultiSense-Viewer.exe"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".myp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt
#define MyAppIcoName "CRL.ico"
#define MyAppSetupIcoName "CRL_Setup.ico"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{AD217912-696A-4087-ACD9-5D396FF870A4}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
ChangesAssociations=yes
DisableProgramGroupPage=yes
; Uncomment the following line to run in non administrative install mode (install for current user only.)
;PrivilegesRequired=lowest
OutputBaseFilename=MultiSenseSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=.\MultiSense-Viewer\Assets\Tools\Windows\{#MyAppSetupIcoName}
UninstallDisplayIcon=.\MultiSense-Viewer\Assets\Tools\Windows\{#MyAppSetupIcoName}
UsePreviousAppDir=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "MultiSense-Viewer\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "MultiSense-Viewer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "MultiSense-Viewer\AutoConnect.exe"; DestDir: "{app}"; Flags: ignoreversion
; NOTE: Don't use "Flags: ignoreversion" on any shared system MultiSense-Viewer

[Registry]
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".myp"; ValueData: ""

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\Assets\Tools\windows\{#MyAppIcoName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\Assets\Tools\windows\{#MyAppIcoName}"; Tasks: desktopicon
[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
Filename: "{sys}\netsh.exe"; Parameters: "firewall add allowedprogram ""{app}\{#MyAppExeName}"" ""MultiSense-Viewer"" ENABLE ALL"; StatusMsg: "MultiSense-Viewer Firewall"; Flags: runhidden; MinVersion: 0,5.01.2600sp2;
Filename: "{sys}\netsh.exe"; Parameters: "firewall add allowedprogram ""{app}\AutoConnect.exe"" ""AutoConnect"" ENABLE ALL"; StatusMsg: "AutoConnect Firewall"; Flags: runhidden; MinVersion: 0,5.01.2600sp2;

; For Windows Vista and later
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""MultiSense-Viewer"" dir=in action=allow program=""{app}\{#MyAppExeName}"" enable=yes"; Flags: runhidden; MinVersion: 0,6.0;
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""AutoConnect"" dir=in action=allow program=""{app}\AutoConnect.exe"" enable=yes"; Flags: runhidden; MinVersion: 0,6.0;

; For Windows XP SP2 and SP3
Filename: "{sys}\netsh.exe"; Parameters: "firewall add allowedprogram ""{app}\{#MyAppExeName}"" ""MultiSense-Viewer"" ENABLE"; Flags: runhidden; OnlyBelowVersion: 0,6.0;
Filename: "{sys}\netsh.exe"; Parameters: "firewall add allowedprogram ""{app}\AutoConnect.exe"" ""AutoConnect"" ENABLE"; Flags: runhidden; OnlyBelowVersion: 0,6.0;