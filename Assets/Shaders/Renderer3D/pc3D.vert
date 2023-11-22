#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec2 inUV1;
layout(location = 4) in vec4 inJoint0;
layout(location = 5) in vec4 inWeight0;
layout(location = 6) in vec4 inColor;

layout (set = 0, binding = 0) uniform UBO
{
    mat4 projectionMatrix;
    mat4 viewMatrix;
    mat4 modelMatrix;
} ubo;

layout (set = 0, binding = 1) uniform PointCloudParam {
    mat4 Q;
    float width;
    float height;
    float disparity;
    float focalLength;
    float scale;
    float pointSize;
} matrix;

layout (set = 0, binding = 2) uniform sampler2D depthMap;


layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 fragPos;


void main()
{

    outUV = inUV;
    gl_PointSize = matrix.pointSize + 1;
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix  * vec4(inPos, 1.0f);
}