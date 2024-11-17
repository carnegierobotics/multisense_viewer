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


layout(set = 2, binding = 0) uniform VertexData {
	vec4 positions[21];
} vertexData;


void main() {
	int idx = gl_VertexIndex;
	// Transform the vertex position to world space
	vec4 worldPos = ubo.model * vec4(vertexData.positions[idx].xyz, 1.0);

	gl_Position = camera.projection * camera.view * worldPos;
}
