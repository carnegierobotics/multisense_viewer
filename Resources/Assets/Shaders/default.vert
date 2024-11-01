#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;

// Camera uniform block
layout (binding = 0) uniform CameraUBO
{
	mat4 projection;
	mat4 view;
	vec3 pos;
} camera;

// Model uniform block
layout (binding = 1) uniform ModelUBO
{
	mat4 model;
} ubo;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 fragPos;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outFragPos; // Pass the world position to fragment shader

void main() {
	// Transform the vertex position to world space
	vec4 worldPos = ubo.model * vec4(inPos, 1.0);
	fragPos = worldPos;

	// Calculate the normal in world space, accounting for non-uniform scaling
	outNormal = mat3(transpose(inverse(ubo.model))) * inNormal;

	// Pass the world space position
	outFragPos = vec3(worldPos);

	// Calculate final position in clip space
	gl_Position = camera.projection * camera.view * worldPos;

	// Pass UV coordinates to fragment shader
	outUV = inUV0;
}
