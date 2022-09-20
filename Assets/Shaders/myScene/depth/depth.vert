#version 450

#define NUM_POINTS 2048

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

layout (binding = 0) uniform UBO
{
    mat4 projectionMatrix;
    mat4 viewMatrix;
    mat4 modelMatrix;
} ubo;

//push constants block
layout(push_constant) uniform constants
{
    vec2 pos;
} MousePos;


layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out vec2 mousePos;


void main()
{
    gl_Position = ubo.modelMatrix  * vec4(inPos, 1.0f);
    outUV = inUV;
    mousePos= MousePos.pos;
}