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

void main() {
	vec4 pos = ubo.model * vec4(inPos, 1.0);
	fragPos = pos;

    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos, 1.0);
    gl_Position = ubo.model * vec4(inPos, 1.0);
    outUV = inUV0;
}

