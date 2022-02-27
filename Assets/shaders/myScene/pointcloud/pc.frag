#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;



void main()
{

    vec3 result = vec3(0.2f, 0.4f, 0.5f); //(ambient + diffuse + specular) * color;
    outColor = vec4(result, 1.0);

}