#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout (push_constant) uniform Material {
    vec2 pos;
} material;


layout(binding = 1, set = 0) uniform Info {
    vec4 lightDir;
    float exposure;
    float gamma;
    float prefilter;
    float ibl;
    float dbgViewInputs;
    vec2 zoomCenter;
    float zoom;
} info;

layout (set = 0, binding = 2) uniform sampler2D samplerColorMap;


void main()
{


    vec3 tex = texture(samplerColorMap, inUV).rgb;
    outColor = vec4(tex.r, tex.r, tex.r, 1.0);

}