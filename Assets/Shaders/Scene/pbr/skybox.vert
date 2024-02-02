#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec2 inUV2;

layout (binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 model;
} ubo;

layout (location = 0) out vec3 outUVW;

// returns a 90 degree x-axis rotation matrix
mat4 get_z_correction_matrix()
{
	float s = sin(radians(90.0));
	float c = cos(radians(90.0));
	return mat4(
	1, 0, 0, 0,
	0, c, s, 0,
	0, -s, c, 0,
	0, 0, 0, 1
	);
}


out gl_PerVertex
{
	vec4 gl_Position;
};

void main()
{
	vec3 pos = inPos;
	mat4 ZUP_CORRECTION = get_z_correction_matrix(); // sused to correct for my z-up view
	mat4 no_translation_view = ubo.view;
	no_translation_view[3] = vec4(0.0, 0.0, 0.0, 1.0); // so skybox doesn't move with camera;
	vec4 outCoords = ubo.projection * no_translation_view * vec4(pos, 1.0);
	outUVW = pos;
	gl_Position = outCoords.xyww;
}
