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
    mat4 Q;
    float width;
    float height;
    float disparity;
} matrix;

layout (set = 0, binding = 2) uniform sampler2D depthMap;


layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out vec3 outCoords;

void main()
{
    gl_PointSize = 1.0f;
    outUV = inUV;
    float width = matrix.width;
    float height = matrix.height;
    // When uploaded to GPU, vulkan will scale the texture values to between 0-1. Since we only have 12 bit values of a 16 bit image, we multiply by 64 to scale between [0 - 1]
    float depth = texture(depthMap, vec2(1-inUV.x, inUV.y)).r * 64;//  scaled to inbetween [0, 1] --

    depth *= 255; // Scale values furter up to 255

    vec4 coords = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    vec2 uvCoords = vec2(int((1 - inUV.x) * width), int((1 - inUV.y) * height));
    if ((uvCoords.x < 20 || uvCoords.x > width - 20) || (uvCoords.y < 20 || uvCoords.y > height - 20) || depth < 5){
        gl_Position = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    vec4 imgCoords = vec4(uvCoords, depth, 1.0f);
    coords = matrix.Q * imgCoords;

    float invB = 1.0f / (-width * depth);
    vec3 outCoordinates = vec3(coords.x * invB, -1 * coords.y * invB, coords.z * invB);

    outCoords = outCoordinates;
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix  * vec4(outCoordinates, 1.0f);
}