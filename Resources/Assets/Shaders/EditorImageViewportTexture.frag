#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) out vec4 outColor;

layout (binding = 1) uniform INFO {
    int selection;
} info;


void main() {
    vec3 value = texture(samplerColorMap, inUV).rgb; // Sample the single channel
    outColor = vec4(value, 1.0f);
}