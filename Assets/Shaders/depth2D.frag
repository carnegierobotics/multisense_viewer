#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(vec3(texture(samplerColorMap, inUV).r), 1.0f);
}