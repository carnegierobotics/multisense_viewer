# install glslc in project folder
git clone https://github.com/google/shaderc
cd shaderc || exit
python3 ./utils/git-sync-deps || exit
mkdir build
cd build || exit
cmake -DCMAKE_BUILD_TYPE=Release .. || exit
make -j12 || exit

echo "Successfully installed glslc compiler in project folder"