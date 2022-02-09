glslc="../compiler/shaderc/build/glslc/glslc"

$glslc ./triangle.vert -o triangle.vert.spv
$glslc ./triangle.frag -o triangle.frag.spv

$glslc ./imgui/ui.vert -o ./imgui/ui.vert.spv
$glslc ./imgui/ui.frag -o ./imgui/ui.frag.spv

$glslc ./gltfLoading/mesh.vert -o ./gltfLoading/mesh.vert.spv
$glslc ./gltfLoading/mesh.frag -o ./gltfLoading/mesh.frag.spv

$glslc ./myScene/box.vert -o ./myScene/box.vert.spv
$glslc ./myScene/box.frag -o ./myScene/box.frag.spv

$glslc ./myScene/sphere/sphere.vert -o ./myScene/sphere/sphere.vert.spv
$glslc ./myScene/sphere/sphere.frag -o ./myScene/sphere/sphere.frag.spv