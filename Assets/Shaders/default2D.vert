#version 450

layout (location = 0) in vec2 inPos;
layout (location = 1) in vec2 inUV0;

layout(location = 0) out vec2 outUV;

void main() {
	gl_Position =  vec4(inPos.xy, 0, 1.0);
    outUV = inUV0;
}

