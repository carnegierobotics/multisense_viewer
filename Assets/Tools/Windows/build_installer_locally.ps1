## Power shell script to automate building a installer. Run this from ./build folder and as admin. Must allow script running on your machine
# Allow scripts to run as admin: Set-ExecutionPolicy RemoteSigned
# Remember to update filepath for installation folder if version number has changed

# Check if path parameter is not specified or empty
# For instance

# ./build_installer_locally.ps1 --path "path-to-project" --version 1.2-0
param(
 [Parameter(Mandatory=$true)]
 [string]$path,

 [Parameter(Mandatory=$true)]
 [string]$version
)


if ([string]::IsNullOrEmpty($path)) {
 Write-Host "Path parameter to CMakeLists file is required."
 exit 1
}

# Your script logic here
Write-Host "You provided the path: $path"
Write-Host "You provided the version: $version"


cmake -B . -DCMAKE_BUILD_TYPE=Release -DWARNINGS_AS_ERRORS=FALSE -DGIT_SUBMODULE=OFF $path
cmake --build . --config Release --target install -- /m:10

mkdir MultiSense-Viewer
Copy-Item -Recurse ".\multisense_${version}_amd64\*" .\MultiSense-Viewer\

mv .\MultiSense-Viewer\Assets\Tools\Windows\inno_setup_script.iss .\ -Force
mv .\MultiSense-Viewer\Assets\Tools\compile.sh .\ -Force
mv .\MultiSense-Viewer\Assets\Tools\install_spirv_compiler.sh .\ -Force
mv .\MultiSense-Viewer\Assets\Tools\monitor_memory_usage.py .\ -Force
mv .\MultiSense-Viewer\Assets\Tools\Windows\build_installer.ps1 .\ -Force
mv .\MultiSense-Viewer\Assets\Tools\Windows\build_installer_locally.ps1 .\ -Force
mv .\MultiSense-Viewer\Assets\Tools\how_to_release.md .\ -Force
rm  -R .\MultiSense-Viewer\Assets\Tools\Ubuntu
rm  -R .\MultiSense-Viewer\bin
rm  -R .\MultiSense-Viewer\include
rm  -R .\MultiSense-Viewer\lib

 & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" .\inno_setup_script.iss