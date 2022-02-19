# install glslc in project folder
mkdir -p "Assets/compiler"
cd "Assets/compiler" || exit
git clone https://github.com/google/shaderc
cd shaderc || exit
python3 ./utils/git-sync-deps || exit
mkdir build
cd build || exit
cmake -DCMAKE_BUILD_TYPE=Release .. || exit
make -j8 || exit

echo "Successfully installed glslc compiler in project folder"