#version 450

#define NUM_POINTS 2048

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

layout (binding = 0) uniform UBO
{
    mat4 projectionMatrix;
    mat4 viewMatrix;
    mat4 modelMatrix;
} ubo;

layout (binding = 1) uniform PointCloudParam {
    mat4 kInverse;
    float width;
    float height;
} matrix;

layout (set = 0, binding = 2) uniform sampler2D depthMap;


layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 fragPos;


void main()
{
    gl_PointSize = 1.0f;
    float width = matrix.width;
    float height = matrix.height;
    float depth = texture(depthMap, inUV).r * 100.0f;

    vec4 coords = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    vec2 uvCoords = vec2(inUV.x * width, inUV.y * height);

    vec4 imgCoords = vec4(uvCoords, 1.0f, 1.0f / depth);
    coords = depth * (matrix.kInverse * imgCoords);
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix  * vec4(coords.xyz, 1.0f);

    outUV = inUV;
}