#version 450


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


struct VertexData {
	vec3 position;   // Aligned to 16 bytes (std430 adds padding)
	vec3 normal;     // Aligned to 16 bytes (std430 adds padding)
	vec2 uv0;        // Aligned to 8 bytes but padded to 16
	vec2 uv1;        // Same alignment rules as uv0
	vec4 color;      // Naturally aligned to 16 bytes
};

layout(std430, set=2, binding=0) readonly buffer CameraGizmoSSBO {
VertexData vertices[];
} vertexData;

// Index buffer SSBO
layout(std430, set=2, binding=1) readonly buffer IndexBufferSSBO {
	uint indices[];
} indexBuffer;


void main() {

	int idx = int(indexBuffer.indices[gl_VertexIndex]);
	vec3 position = vertexData.vertices[idx].position;
	// Transform the vertex to world space and then to clip space
	vec4 worldPos = ubo.model * vec4(position, 1.0);
	gl_Position = camera.projection * camera.view * worldPos;
}