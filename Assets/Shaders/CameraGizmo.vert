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
	vec4 position;   // Aligned to 16 bytes (std430 adds padding)
	vec4 color;      // Naturally aligned to 16 bytes
};

layout(std430, set=2, binding=0) readonly buffer CameraGizmoSSBO {
VertexData vertices[];
} vertexData;

// Index buffer SSBO
layout(std430, set=2, binding=1) readonly buffer IndexBufferSSBO {
	uint indices[];
} indexBuffer;


layout(location = 0) out vec4 outColor;

void main() {
	int idx = int(indexBuffer.indices[gl_VertexIndex]);
	vec3 position = vertexData.vertices[idx].position.xyz;
	outColor = vertexData.vertices[idx].color;

	// Transform the vertex to world space and then to clip space
	vec4 worldPos = ubo.model * vec4(position, 1.0f);
	gl_Position = camera.projection * camera.view * worldPos;
}