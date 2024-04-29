#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

// Shared set between most vertex shaders
layout(set = 0, binding = 0) uniform ViewUniforms {
    mat4 projection;
    mat4 view;
    mat4 model;
    vec3 camPos;
} view;

layout(location = 0) out float near;
layout(location = 1) out float far;
layout(location = 2) out vec3 nearPoint;
layout(location = 3) out vec3 farPoint;
layout(location = 4) out mat4 fragProj;
layout(location = 8) out mat4 fragView;

vec3 UnprojectPoint(float x, float y, float z, mat4 view, mat4 projection) {
    mat4 viewInv = inverse(view);
    mat4 projInv = inverse(projection);
    vec4 unprojectedPoint =  viewInv * projInv * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

// normal vertice projection
void main() {
    near = 0.1;
    far = 100;
    fragProj = view.projection;
    fragView = view.view;
    vec3 p = inPos;
    nearPoint = UnprojectPoint(p.x, p.y, 0.0, view.view, view.projection).xyz; // unprojecting on the near plane
    farPoint = UnprojectPoint(p.x, p.y, 1.0, view.view, view.projection).xyz; // unprojecting on the far plane
    gl_Position = vec4(p, 1.0); // using directly the clipped coordinates
}