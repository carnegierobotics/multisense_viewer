#version 450

layout (location = 0) in vec2 inUV;

layout(binding = 1) uniform Info {
    vec4 lightDir;
    vec4 zoom;
    vec4 zoomTranslate;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float debugViewInputs;
    float lod;
    vec2 pad;
    vec4 normalizeVal;
    vec4 kernelFilters;
} info;

layout (binding = 2) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

void main()
{
	vec2 uv = inUV;
	outColor = texture(samplerColorMap, uv);
}