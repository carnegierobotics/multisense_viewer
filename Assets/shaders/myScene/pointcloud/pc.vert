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

layout (binding = 1) uniform PointCloud
{
    vec4 pos[NUM_POINTS];
    vec4 col[NUM_POINTS];
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragPos;


void main()
{
    gl_PointSize = 10.0f;
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * pc.pos[gl_VertexIndex];

}