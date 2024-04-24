#version 450

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
    vec3 camPos;
} ubo;

layout (location = 0) out vec4 outColor;

void main() {

    outColor = vec4(vec3(1.0f), 1.0f);
}