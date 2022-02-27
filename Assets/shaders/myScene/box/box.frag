#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 FragPos;
layout (location = 4) in mat4 model;

layout(set = 0, binding = 1) uniform INFO {
    vec4 lightColor;
    vec4 objectColor;
    vec4 lightPos;
    vec4 viewPos;
} info;

layout(set = 0, binding = 2) uniform SELECT {
    float map;
} select ;


layout (location = 0) out vec4 outFragColor;

void main()
{

    outFragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);

}