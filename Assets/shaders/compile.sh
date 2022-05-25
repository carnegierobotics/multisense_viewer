#!/bin/bash

# Simple script to compile shaders. Just configure it to run on every build for convenience.
# Make sure the glslc exec is installed.

glslc="../compiler/shaderc/build/glslc/glslc" # usual linux path
#glslc="../../compiler/shaderc/build/glslc/Debug/glslc" # usual windows path

sceneOutDir="./myScene/spv/"
sceneDir="./myScene/"

mkdir -p ${sceneOutDir}

$glslc ${sceneDir}triangle/triangle.vert -o ${sceneOutDir}triangle.vert.spv
$glslc ${sceneDir}triangle/triangle.frag -o ${sceneOutDir}triangle.frag.spv

$glslc ./imgui/ui.vert -o ./imgui/ui.vert.spv
$glslc ./imgui/ui.frag -o ./imgui/ui.frag.spv

# $glslc ${sceneDir}/fox/fox.vert -o ${sceneOutDir}fox.vert.spv
# $glslc ${sceneDir}/fox/fox.frag -o ${sceneOutDir}fox.frag.spv

$glslc ${sceneDir}/helmet/helmet.vert -o ${sceneOutDir}helmet.vert.spv
$glslc ${sceneDir}/helmet/helmet.frag -o ${sceneOutDir}helmet.frag.spv

$glslc ${sceneDir}/box/box.vert -o ${sceneOutDir}box.vert.spv
$glslc ${sceneDir}/box/box.frag -o ${sceneOutDir}box.frag.spv

$glslc ${sceneDir}sphere/sphere.vert -o ${sceneOutDir}sphere.vert.spv
$glslc ${sceneDir}sphere/sphere.frag -o ${sceneOutDir}sphere.frag.spv

$glslc ${sceneDir}pointcloud/pc.vert -o ${sceneOutDir}pointcloud.vert.spv
$glslc ${sceneDir}pointcloud/pc.frag -o ${sceneOutDir}pointcloud.frag.spv

$glslc ${sceneDir}quad/quad.vert -o ${sceneOutDir}quad.vert.spv
$glslc ${sceneDir}quad/quad.frag -o ${sceneOutDir}quad.frag.spv

$glslc ${sceneDir}depth/depth.vert -o ${sceneOutDir}depth.vert.spv
$glslc ${sceneDir}depth/depth.frag -o ${sceneOutDir}depth.frag.spv

$glslc ${sceneDir}depth/preview.vert -o ${sceneOutDir}preview.vert.spv
$glslc ${sceneDir}depth/preview.frag -o ${sceneOutDir}preview.frag.spv