#!/bin/bash

# Check if the script is run with one argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory-name> (Usually something like: multisense_1.1-0_amd64)"
    exit 1
fi

# Define the build directory based on the argument
BUILD_DIR=$1

# Reminder to run from the correct directory
echo "Make sure to run this from a {source}/build folder"

# Run cmake commands
cmake .. -DCMAKE_BUILD_TYPE=Release  -DWARNINGS_AS_ERRORS=FALSE && cmake --build . --config Release --target install -- -j32

# Create necessary directories
mkdir -p $BUILD_DIR/DEBIAN
mkdir -p $BUILD_DIR/usr/share/applications
mkdir -p $BUILD_DIR/opt/multisense/

# Copy necessary files
cp ../Assets/Tools/Ubuntu/start.sh $BUILD_DIR/opt/multisense/
cp ../Assets/Tools/Ubuntu/multisense.desktop $BUILD_DIR/usr/share/applications/
cp ../Assets/Tools/Ubuntu/control $BUILD_DIR/DEBIAN

# Remove unnecessary directories
rm -rf $BUILD_DIR/opt/multisense/include
rm -rf $BUILD_DIR/opt/multisense/lib
rm -rf $BUILD_DIR/opt/multisense/share
rm -rf $BUILD_DIR/opt/multisense/bin
rm -rf $BUILD_DIR/opt/multisense/Assets/Tools

# Build the Debian package
dpkg-deb --build --root-owner-group $BUILD_DIR

echo "Package built successfully: $BUILD_DIR.deb"
