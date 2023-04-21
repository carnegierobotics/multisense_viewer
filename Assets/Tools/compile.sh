#!/bin/bash

# Simple script to compile shaders. Just configure it to run on every build for convenience.
# Make sure the glslc exec is installed.

# Windows location
outDir="./../../cmake-build-debug/Assets/Shaders/Scene/spv/"
glslc="../../shaderc/build/glslc/Debug/glslc" #

# Unix location
#glslc="../../shaderc/build/glslc/glslc" #

sceneOutDir="../Shaders/Scene/spv/"
sceneDir="../Shaders/Scene"

mkdir -p ${sceneOutDir}

$glslc ${sceneDir}/video/color.vert -o ${sceneOutDir}color.vert.spv
$glslc ${sceneDir}/video/disparity.vert -o ${sceneOutDir}disparity.vert.spv
$glslc ${sceneDir}/video/grayscale.vert -o ${sceneOutDir}grayscale.vert.spv
$glslc ${sceneDir}/video/color_default_sampler.frag -o ${sceneOutDir}color_default_sampler.frag.spv
$glslc ${sceneDir}/video/color_ycbcr_sampler.frag -o ${sceneOutDir}color_ycbcr_sampler.frag.spv
$glslc ${sceneDir}/video/disparity.frag -o ${sceneOutDir}disparity.frag.spv
$glslc ${sceneDir}/video/grayscale.frag -o ${sceneOutDir}grayscale.frag.spv

echo "Compiled video shaders"
$glslc ${sceneDir}/pointcloud/pc.vert -o ${sceneOutDir}pointcloud.vert.spv
$glslc ${sceneDir}/pointcloud/pc.frag -o ${sceneOutDir}pointcloud.frag.spv
echo "Compiled pointcloud shaders"

$glslc  ${sceneDir}/imgui/ui.vert -o ${sceneOutDir}/ui.vert.spv
$glslc  ${sceneDir}/imgui/ui.frag -o ${sceneOutDir}/ui.frag.spv
echo "Compiled UI shaders"
$glslc ${sceneDir}/pbr/object.vert -o ${sceneOutDir}object.vert.spv
$glslc ${sceneDir}/pbr/object.frag -o ${sceneOutDir}object.frag.spv
$glslc ${sceneDir}/pbr/skybox.vert -o ${sceneOutDir}skybox.vert.spv
$glslc ${sceneDir}/pbr/skybox.frag -o ${sceneOutDir}skybox.frag.spv
$glslc ${sceneDir}/pbr/genbrdflut.vert -o ${sceneOutDir}genbrdflut.vert.spv
$glslc ${sceneDir}/pbr/genbrdflut.frag -o ${sceneOutDir}genbrdflut.frag.spv
$glslc ${sceneDir}/pbr/filtercube.vert -o ${sceneOutDir}filtercube.vert.spv
$glslc ${sceneDir}/pbr/irradiancecube.frag -o ${sceneOutDir}irradiancecube.frag.spv
$glslc ${sceneDir}/pbr/prefilterenvmap.frag -o ${sceneOutDir}prefilterenvmap.frag.spv
echo "Compiled PBR shaders"



echo "Copying to debug build location"
cp ${sceneOutDir}/*.spv ${outDir}

echo "Press any key to exit..."
read -n 1 -s -r -p ""

echo "Exiting..."