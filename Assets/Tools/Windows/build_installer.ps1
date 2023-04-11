## Power shell script to automate building a installer. Run this from ./build folder and as admin

cmake -B . -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE=OFF ..
cmake --build . --config Release --target install

mkdir files
Copy-Item -Recurse .\multisense_1.0-0_amd64\* .\files\

mv .\files\Assets\Tools\Windows\inno_setup_script.iss .\
mv .\files\Assets\Tools\compile.sh .\
mv .\files\Assets\Tools\install_spirv_compiler.sh .\
mv .\files\Assets\Tools\monitor_memory_usage.py .\
mv .\files\Assets\Tools\build_installer.ps1 .\
mv .\files\Assets\Tools\Windows\build_installer.ps1 .\
rm  -R .\files\Assets\Tools\Ubuntu
rm  -R .\files\bin
rm  -R .\files\include
rm  -R .\files\lib

 & "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" .\inno_setup_script.iss