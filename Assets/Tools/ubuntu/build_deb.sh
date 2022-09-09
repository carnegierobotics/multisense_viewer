# Run from build folder
echo "Make sure to run this from a {source}/build folder"
mkdir multisense_1.0-0_amd64/DEBIAN
mkdir -p multisense_1.0-0_amd64/usr/share/applications
cp ../Assets/Tools/ubuntu/start.sh multisense_1.0-0_amd64/opt/multisense/
cp ../Assets/Tools/ubuntu/multisense.desktop multisense_1.0-0_amd64/usr/share/applications/
cp ../Assets/Tools/ubuntu/control multisense_1.0-0_amd64/DEBIAN
dpkg-deb --build multisense_1.0-0_amd64