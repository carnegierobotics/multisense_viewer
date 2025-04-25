#version 450

#define NUM_POINTS 2048

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;

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
} model;

layout (set = 1, binding = 0) uniform PointCloudParam {
    mat4 Q;
    mat4 colorIntrinsics;
    mat4 colorExtrinsics;
    float width;
    float height;
    float disparity;
    float focalLength;
    float scale;
    float pointSize;
    float useColor;
    float hasSampler;
} matrix;

layout (set = 1, binding = 1) uniform sampler2D disparityImage;


layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 fragPos;
layout(location = 3) out vec3 outCoords;


void main()
{


    outUV = inUV;
    float width = matrix.width;
    float height = matrix.height;
    // When uploaded to GPU, vulkan will scale the texture values to between 0-1. Since we only have 12 bit values of a 16 bit image, we multiply by 64 to scale between [0 - 1]
    float depth = texture(disparityImage, vec2(inUV.x, inUV.y)).r * 64 * 255;//  scaled to inbetween [0, 1] then up to 0 - 255 for disparity

    gl_PointSize = matrix.pointSize;

    vec2 uvCoords = vec2(int((inUV.x) * width), int((inUV.y) * height));
    if ((uvCoords.x < 20 || uvCoords.x > width - 20) || (uvCoords.y < 20 || uvCoords.y > height - 20) || depth < 5){
        gl_Position = vec4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    float invB = matrix.focalLength / depth;

    vec4 imgCoords = vec4(uvCoords, depth, 1);
    vec4 coords = matrix.Q * imgCoords * (1/invB);
    coords = coords / coords.w * matrix.scale;
    vec3 outCoordinates = vec3(coords.x, coords.y, coords.z);


    // Replace ubo.viewMatrix with viewMatrix in your gl_Position calculation
    gl_Position = camera.projection * camera.view * model.model * vec4(outCoordinates, 1.0f);


    outCoords = outCoordinates;
    //gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix  * vec4(outCoordinates, 1.0f);
    //gl_Position = ubo.viewMatrix * ubo.modelMatrix  * vec4(outCoordinates, 1.0f);
}