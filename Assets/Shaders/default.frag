#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 fragPos;  // Assuming fragPos is passed from vertex shader

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
    vec3 camPos;
} ubo;

layout(binding = 1) uniform Info {
    vec4 lightDir;
    vec4 zoom;

} info;

layout (binding = 2) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

void main()
{
    float depth = length(ubo.camPos) - length(fragPos);

    outColor = texture(samplerColorMap, inUV);
    //outColor = vec4(vec3(depth), 1.0);
}