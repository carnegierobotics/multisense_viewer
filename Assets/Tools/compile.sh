#!/bin/bash

# Simple script to compile shaders. Just configure it to run on every build for convenience.
# Make sure the glslc exec is installed.
# Run compile.sh windows  for windows

if [[ "$1" == "windows" ]]; then
    # Windows location
    echo "Compiling from Windows: cwd: $(pwd)"
    # shellcheck disable=SC2046
    project_path="$(pwd)/../.."
    echo "Compiling from Windows: project path: ${project_path})"
    outDir="${project_path}/cmake-build-debug/Assets/Shaders/Scene/spv/"
    glslc="${project_path}/shaderc/build/glslc/Debug/glslc.exe"
    sceneOutDir="${project_path}/Assets/Shaders/Scene/spv/"
    sceneDir="${project_path}/Assets/Shaders/Scene"
    outDir="${project_path}/cmake-build-debug/Assets/Shaders/Scene/spv/"

        echo "compiling shader from folder ${sceneDir}"
        echo "copying to this folder ${sceneOutDir}"
        echo "additional copy to ${outDir}"

else
    # Unix location
    echo "Compiling from ubuntu $(pwd)"
    glslc="../../shaderc/build/glslc/glslc"
    outDir="./../../cmake-build-debug/Assets/Shaders/Scene/spv/"
    sceneOutDir="../Shaders/Scene/spv/"
    sceneDir="../Shaders/Scene"
    renderer3DDir="../Shaders/Renderer3D"
fi


mkdir -p ${sceneOutDir}

#$glslc ${sceneDir}/video/color.vert -o ${sceneOutDir}color.vert.spv
#$glslc ${sceneDir}/video/disparity.vert -o ${sceneOutDir}disparity.vert.spv
$glslc ${sceneDir}/video/grayscale.vert -o ${sceneOutDir}grayscale.vert.spv
#$glslc ${sceneDir}/video/color_default_sampler.frag -o ${sceneOutDir}color_default_sampler.frag.spv
#$glslc ${sceneDir}/video/color_ycbcr_sampler.frag -o ${sceneOutDir}color_ycbcr_sampler.frag.spv
#$glslc ${sceneDir}/video/disparity.frag -o ${sceneOutDir}disparity.frag.spv
$glslc ${sceneDir}/video/grayscale.frag -o ${sceneOutDir}grayscale.frag.spv
$glslc ${sceneDir}/video/compute.frag -o ${sceneOutDir}compute.frag.spv

#echo "Compiled video shaders"
#$glslc ${sceneDir}/pointcloud/pc.vert -o ${sceneOutDir}pointcloud.vert.spv
#$glslc ${sceneDir}/pointcloud/pc.frag -o ${sceneOutDir}pointcloud.frag.spv
#echo "Compiled pointcloud shaders"

#$glslc  ${sceneDir}/imgui/ui.vert -o ${sceneOutDir}ui.vert.spv
#$glslc  ${sceneDir}/imgui/ui.frag -o ${sceneOutDir}ui.frag.spv
#echo "Compiled UI shaders"
#$glslc ${sceneDir}/pbr/object.vert -o ${sceneOutDir}object.vert.spv
#$glslc ${sceneDir}/pbr/object.frag -o ${sceneOutDir}object.frag.spv
#$glslc ${sceneDir}/pbr/skybox.vert -o ${sceneOutDir}skybox.vert.spv
#$glslc ${sceneDir}/pbr/skybox.frag -o ${sceneOutDir}skybox.frag.spv
#$glslc ${sceneDir}/pbr/genbrdflut.vert -o ${sceneOutDir}genbrdflut.vert.spv
#$glslc ${sceneDir}/pbr/genbrdflut.frag -o ${sceneOutDir}genbrdflut.frag.spv
#$glslc ${sceneDir}/pbr/filtercube.vert -o ${sceneOutDir}filtercube.vert.spv
#$glslc ${sceneDir}/pbr/irradiancecube.frag -o ${sceneOutDir}irradiancecube.frag.spv
#$glslc ${sceneDir}/pbr/prefilterenvmap.frag -o ${sceneOutDir}prefilterenvmap.frag.spv
#echo "Compiled PBR shaders"

#$glslc ${renderer3DDir}/grid.vert -o ${sceneOutDir}grid.vert.spv
#$glslc ${renderer3DDir}/grid.frag -o ${sceneOutDir}grid.frag.spv
#$glslc ${renderer3DDir}/pc3D.vert -o ${sceneOutDir}pc3D.vert.spv
#$glslc ${renderer3DDir}/pc3D.frag -o ${sceneOutDir}pc3D.frag.spv
#echo "Compiled Renderer3D shaders"

$glslc ${sceneDir}/stereo_sim.comp -o ${sceneOutDir}stereo_sim.comp.spv
$glslc ${sceneDir}/stereo_sim_ext.comp -o ${sceneOutDir}stereo_sim_ext.comp.spv
$glslc ${sceneDir}/stereo_sim_pix_norm_pass.comp -o ${sceneOutDir}stereo_sim_pix_norm_pass.comp.spv
$glslc ${sceneDir}/stereo_sim_pass_2.comp -o ${sceneOutDir}stereo_sim_pass_2.comp.spv
$glslc ${sceneDir}/stereo_sim_pass_3.comp -o ${sceneOutDir}stereo_sim_pass_3.comp.spv
$glslc ${sceneDir}/stereo_sim_pass_4.comp -o ${sceneOutDir}stereo_sim_pass_4.comp.spv
$glslc ${sceneDir}/particle.frag -o ${sceneOutDir}particle.frag.spv
$glslc ${sceneDir}/particle.vert -o ${sceneOutDir}particle.vert.spv
echo "Compiled Compute shaders"


echo "Copying to debug build location: ${sceneOutDir}*.spv | to | ${outDir}"
cp ${sceneOutDir}*.spv  ${outDir}
if [[ "$1" == "windows" ]]; then
  echo "Press any key to exit..."
  read -n 1 -s -r -p ""
fi

echo "Exiting..."