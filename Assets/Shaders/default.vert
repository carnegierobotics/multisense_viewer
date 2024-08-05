#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;

layout (binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 model;
	vec3 camPos;
} ubo;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 fragPos;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outFragPos; // Pass the world position to fragment shader

void main() {
	vec4 pos = ubo.model * vec4(inPos, 1.0);
	fragPos = pos;
	mat4 v = ubo.view;
	mat4 p = ubo.projection;
	outNormal = mat3(transpose(inverse(ubo.model))) * inNormal; // Correct normal transformation
	outFragPos = vec3(pos); // World space position
	gl_Position = p * v * pos;
	outUV = inUV0;
}
