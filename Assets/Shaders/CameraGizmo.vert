#version 450


layout (binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 model;
	vec3 camPos;
} ubo;

layout(binding = 1) uniform VertexData {
	vec4 positions[21];
} vertexData;

void main() {
	int idx = gl_VertexIndex;

	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vertexData.positions[idx].xyz, 1.0);
}

