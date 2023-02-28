#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec3 inCoords;


layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;

layout (set = 0, binding = 4) uniform PointCloudParam {
    mat4 intrinsic;
    mat4 extrinsic;
} mat;

layout(location = 0) out vec4 outColor;


void main()
{
    vec2 uv = vec2(1-inUV.x, inUV.y);
    // Project into color image
    vec4 uvColorProjected = 1/inCoords.z * mat.intrinsic * mat.extrinsic * vec4(inCoords, 1.0f);

    // Sample bilinearly

    vec3 tex = texture(samplerColorMap, vec2(uvColorProjected.rg)).rgb;
    outColor = vec4(tex, 1.0);

}