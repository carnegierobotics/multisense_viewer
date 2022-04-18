#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;


void main()
{


    outColor = vec4(0.5, 0.5, 0.5, 1.0);

}