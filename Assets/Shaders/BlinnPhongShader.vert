#version 450
/* ---------- per-vertex (binding 0) ---------- */
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec2 inUV2;
layout(location = 4) in vec4 inColor;

/* global UBO set=0 binding=0 unchanged        */
layout(set = 0, binding = 0) uniform GlobalUBO {
	mat4 view;
	mat4 proj;
	vec3 cameraPos;
	float numLights;
	vec4 lightPos[16];
} uGlobal;

/* transform SSBO set=1 binding=0              */
layout(set = 1, binding = 0) readonly buffer TransformBuf {
	mat4 model[];
} uTransforms;

/* push-constants                              */
layout(push_constant) uniform BatchPC {
	uint transformBase;
	uint materialBase;
} pc;

/* outputs */
layout(location=0) out vec2 vUV;
layout(location=1) out vec3 vNormal;
layout(location=2) out vec3 vWorldPos;
/* you can still pass the instance index if you like */
layout(location=3) flat out uint vInstIdx;

void main() {
	uint idx   = pc.transformBase + gl_InstanceIndex;
	mat4 M     = uTransforms.model[idx];

	vec4 world = M * vec4(inPos,1.0);
	vWorldPos  = world.xyz;
	vNormal    = mat3(transpose(inverse(M))) * inNormal;
	vUV        = inUV;
	vInstIdx   = gl_InstanceIndex;

	gl_Position = uGlobal.proj * uGlobal.view * world;
}
