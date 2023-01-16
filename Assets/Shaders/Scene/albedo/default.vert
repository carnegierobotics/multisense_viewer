#version 450

#define NUM_POINTS 2048

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec2 inUV2;

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 fragPos;

void main()
{
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);
    outUV = inUV;
}