#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

layout (set = 0, binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 model;
	vec3 camPos;
} ubo;

layout(location = 0) out vec2 outUV;


void main() {
	vec4  locPos = ubo.model * vec4(inPos, 1.0);
	vec3 outWorldPos = locPos.xyz / locPos.w;

	outUV = inUV;
	gl_Position =  ubo.projection * ubo.view * vec4(outWorldPos, 1.0);

}