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


cmake -B . -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE=OFF $path
cmake --build . --config Release --target install -- /m:10

mkdir files
Copy-Item -Recurse ".\multisense_${version}_amd64\*" .\files\

mv .\files\Assets\Tools\Windows\inno_setup_script.iss .\
mv .\files\Assets\Tools\compile.sh .\
mv .\files\Assets\Tools\install_spirv_compiler.sh .\
mv .\files\Assets\Tools\monitor_memory_usage.py .\
mv .\files\Assets\Tools\Windows\build_installer.ps1 .\
mv .\files\Assets\Tools\Windows\build_installer_locally.ps1 .\
mv .\files\Assets\Tools\how_to_release .\
rm  -R .\files\Assets\Tools\Ubuntu
rm  -R .\files\bin
rm  -R .\files\include
rm  -R .\files\lib

 & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" .\inno_setup_script.iss