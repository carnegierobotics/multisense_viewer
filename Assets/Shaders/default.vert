#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;
layout (location = 3) in vec2 inUV1;
layout (location = 4) in vec4 inJoint0;
layout (location = 5) in vec4 inWeight0;
layout (location = 6) in vec4 inColor0;

layout (set = 0, binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 model;
	vec3 camPos;
} ubo;

layout(location = 0) out vec2 outUV;

void main() {
    //gl_Position = positions[gl_VertexIndex];
    //outUV = texpos[gl_VertexIndex];
    //vec4 pos = ubo.model * vec4(inPos, 1.0f);
    //gl_Position =  ubo.projection * ubo.view * vec4(pos.xyz, 1.0);

    gl_Position = vec4(inPos, 1.0);
    outUV = inUV0;

}

