#version 450

/* ---------- per-vertex inputs (binding 0) ---------- */
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV0;
layout(location = 3) in vec2 inUV1;
layout(location = 4) in vec4 inColor;

/* ---------- per-instance inputs (binding 1) ---------- */
layout(location = 5) in vec4 iModel0;   // first column of model matrix
layout(location = 6) in vec4 iModel1;
layout(location = 7) in vec4 iModel2;
layout(location = 8) in vec4 iModel3;


/* ---------- camera block (once per draw call) ------ */
layout(binding = 0) uniform CameraUBO {
	mat4 projection;
	mat4 view;
	vec3 camPos;
} camera;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 fragPos;
layout(location = 2) out vec3 outNormal;

void main()
{
/* reassemble the mat4 from the four attributes */
	mat4 model = mat4(iModel0, iModel1, iModel2, iModel3);

	vec4 worldPos = model * vec4(inPos, 1.0);
	fragPos      = worldPos;
	outNormal    = mat3(transpose(inverse(model))) * inNormal;

	gl_Position  = camera.projection * camera.view * worldPos;
	outUV        = inUV0;
}