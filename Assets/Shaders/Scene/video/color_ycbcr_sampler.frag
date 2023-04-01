#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 0) uniform Info {
    vec4 lightDir;
    vec4 zoom;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float debugViewInputs;
    float lod;
    vec2 pad;
} info;

layout (set = 0, binding = 2) uniform sampler2D luma;

layout (set = 0, binding = 3) uniform sampler2D chromaU;

layout (set = 0, binding = 4) uniform sampler2D chromaV;


void main()
{
    vec2 zoom = vec2(info.zoom.x, info.zoom.y);

    float uvSampleX = (inUV.x + zoom.x - info.zoom.w) / info.zoom.z + info.zoom.w;
    float uvSampleY = (inUV.y - zoom.y) / info.zoom.z + zoom.y;
    vec2 uv = vec2(uvSampleX, uvSampleY);
    float r, g, b, y, u, v;
    mat3 colorMatrix = mat3(
                1,   1,       1,
                0,  -0.344,  1.72,
                1.402,   -0.71,   0);
    y = texture(luma, uv).r;
    u = texture(chromaU, uv).r - 0.5f;
    v = texture(chromaV, uv).r - 0.5f;
    vec3 yuv = vec3(y, u, v);
    outColor = vec4(colorMatrix * yuv, 1.0f);
}