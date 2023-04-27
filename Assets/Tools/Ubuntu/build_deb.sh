# Run from build folder
echo "Make sure to run this from a {source}/build folder"
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release --target install -- -j12
mkdir multisense_1.1-0_amd64/DEBIAN
mkdir -p multisense_1.1-0_amd64/usr/share/applications
cp ../Assets/Tools/Ubuntu/start.sh multisense_1.1-0_amd64/opt/multisense/
cp ../Assets/Tools/Ubuntu/multisense.desktop multisense_1.1-0_amd64/usr/share/applications/
cp ../Assets/Tools/Ubuntu/control multisense_1.1-0_amd64/DEBIAN
rm -rf multisense_1.1-0_amd64/opt/multisense/include
rm -rf multisense_1.1-0_amd64/opt/multisense/lib
rm -rf multisense_1.1-0_amd64/opt/multisense/share
rm -rf multisense_1.1-0_amd64/opt/multisense/bin
rm -rf multisense_1.1-0_amd64/opt/multisense/Assets/Tools
dpkg-deb --build --root-owner-group multisense_1.1-0_amd64