#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 0) uniform Colors {
    vec4 objectColor;
    vec4 lightColor;
    vec4 lightPos;
    vec4 viewPos;
} colors;

layout(set = 0, binding = 2) uniform SELECT {
    float map;
} select ;

layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;


void main()
{

    vec3 tex = texture(samplerColorMap, inUV).rgb * 65535 / 4096 * 2;

    outColor = vec4(tex.r, tex.r, tex.r, 1.0);

}