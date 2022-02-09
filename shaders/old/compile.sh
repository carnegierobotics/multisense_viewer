glslc ./defaultVert.vert -o defaultVert.spv
glslc ./defaultFrag.frag -o defaultFrag.spv
glslc ./PhongLight.frag -o phongLightFrag.spv
glslc ./lamp.vert -o lampVert.spv
glslc ./lamp.frag -o lampFrag.spv

glslc ./experimental/computeShader.comp -o ./experimental/computeShader.spv
glslc ./experimental/computeDisparity.comp -o ./experimental/computeDisparity.spv
glslc ./experimental/copyShader.comp -o ./experimental/copyShader.spv


glslc ./textoverlay/text.vert -o ./textoverlay/textVert.spv
glslc ./textoverlay/text.frag -o ./textoverlay/textFrag.spv
glslc ./textoverlay/ui.vert -o ./textoverlay/ui.vert.spv
glslc ./textoverlay/ui.frag -o ./textoverlay/ui.frag.spv