#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform Colors {
    vec4 objectColor;
    vec4 lightColor;
    vec4 lightPos;
    vec4 viewPos;
} colors;

layout (set = 0, binding = 2) uniform sampler2D luma;


layout (set = 0, binding = 3) uniform sampler2D chromaU;

layout (set = 0, binding = 4) uniform sampler2D chromaV;


void main()
{

    float r, g, b, y, u, v;


    r = y + 1.13983*v;
    g = y - 0.39465*u - 0.58060*v;
    b = y + 2.03211*u;


    outColor =  vec4(r, g, b, 1.0);

}