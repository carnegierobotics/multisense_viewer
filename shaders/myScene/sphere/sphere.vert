#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;

layout (set = 0, binding = 0) uniform UBOScene
{
    mat4 projection;
    mat4 view;
    mat4 model;
} ubo;

layout (location = 0) out vec3 outPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 FragPos;
layout (location = 4) out mat4 model;

void main()
{

    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);

    outNormal = mat3(transpose(inverse(ubo.model))) * inNormal;
    outUV = inUV;
    FragPos = vec3(ubo.model * vec4(inPos, 1.0));

    model = ubo.model;
    outPos = inPos;

}