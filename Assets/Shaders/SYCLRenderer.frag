#version 450

layout (location = 0) in vec2 inUV;

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


void main() {

    outColor = vec4(texture(samplerColorMap, vec2(inUV.x, 1-inUV.y)).rgb, 1.0f);
}