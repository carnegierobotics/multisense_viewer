#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;
layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;


layout(location = 0) out vec4 outColor;


void main()
{
    vec3 tex = texture(samplerColorMap, vec2(1-inUV.x, inUV.y)).rgb;
    outColor = vec4(tex.r, tex.r, tex.r, 1.0);

}