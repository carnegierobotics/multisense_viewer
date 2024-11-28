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
	// Fetch the index from the index buffer using gl_VertexIndex
	int idx = int(indexBuffer.indices[gl_VertexIndex]);

	// Get the vertex data using the fetched index
	vec3 position = vertexData.vertices[idx].position;

	// Transform the vertex position to world space
	vec4 worldPos = ubo.model * vec4(position, 1.0f);

	// Compute the final clip-space position
	gl_Position = camera.projection * camera.view * worldPos;
}