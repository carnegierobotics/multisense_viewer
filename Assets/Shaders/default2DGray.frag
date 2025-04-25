#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) out vec4 outColor;

void main() {
    vec4 color =  texture(samplerColorMap, inUV);

    outColor = vec4(color.r, color.r, color.r, 1.0);
}