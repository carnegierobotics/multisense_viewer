#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(set = 0, binding = 0) uniform UboViewProjection {
    mat4 projection;
    mat4 view;
    mat4 model;
} uboViewProjection;


void main() {
    gl_Position = uboViewProjection.projection * uboViewProjection.view * uboViewProjection.model * vec4(inPosition, 1.0);

}