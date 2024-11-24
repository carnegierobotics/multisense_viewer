#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) out vec4 outColor;

layout (binding = 1) uniform INFO {
    int selection;
} info;


void main() {
    float value = texture(samplerColorMap, inUV).r; // Sample the single channel
    vec3 finalColor = vec3(value);                 // Expand to RGB if needed
    // Use finalColor for further processing
    outColor = vec4(finalColor, 1.0f);
}