#version 450

layout (binding = 1) uniform sampler2D texSampler;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

void main()
{
	vec2 uv = inUV;
	uv.y = 1 - uv.y;
	outColor = texture(texSampler, uv);
}