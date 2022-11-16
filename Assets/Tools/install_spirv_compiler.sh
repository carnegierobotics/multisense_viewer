# install glslc in project folder
git clone https://github.com/google/shaderc ../../
cd ../../shaderc || exit
python3 ./utils/git-sync-deps || exit
mkdir build
cd build || exit
cmake -DCMAKE_BUILD_TYPE=Release .. || exit
make -j12 || exit

echo "Successfully installed glslc compiler in project folder"

# IF ON WINDOWS
# Open developer command prompt. Min version MSVC 2015
# cd $BUILD_DIR
# cmake $SOURCE_DIR
# cmake --build . --config {Release|Debug|MinSizeRel|RelWithDebInfo}
# ctest -C {Release|Debug|MinSizeRel|RelWithDebInfo}