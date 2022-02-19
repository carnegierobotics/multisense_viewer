#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

layout (binding = 0) uniform UBO
{
	mat4 projectionMatrix;
	mat4 viewMatrix;
	mat4 modelMatrix;
} ubo;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragPos;



void main()
{
	gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(inPos.xyz, 1.0);

	fragTexCoord = inUV;
	fragPos = vec3(ubo.modelMatrix * vec4(inPos, 1.0));
	outNormal = mat3(transpose(inverse(ubo.modelMatrix))) * inNormal;

}