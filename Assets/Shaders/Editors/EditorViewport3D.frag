#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(samplerColorMap, inUV).rgb;
    outColor = vec4(color, 1.0);
}