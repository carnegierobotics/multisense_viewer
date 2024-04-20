#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 fragPos;  // Assuming fragPos is passed from vertex shader

layout (binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
    vec3 camPos;
} ubo;

layout(binding = 1) uniform Info {
    vec4 lightDir;
    vec4 zoom;

} info;

layout (binding = 2) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

void main()
{
    // Perspective divide to get normalized device coordinates
    vec3 ndcPos = fragPos.xyz / fragPos.w;
    float depth = ndcPos.z + 0.5;  // Depth is now in the range [0, 1]

    // Extract the near and far plane values from the projection matrix
    float near = -ubo.projection[3][2] / (ubo.projection[2][2] - 1.0);
    float far = ubo.projection[3][2] / (1.0 + ubo.projection[2][2]);

    // Convert from NDC depth to eye space depth
    float eyeDepth = (2.0 * near * far) / (far + near - depth * (far - near));

    // Calculate disparity as some function of eyeDepth; example: inverse depth
    float disparity = 1.0 / depth;

    // Visualize disparity as grayscale (normalized based on expected disparity range)
    float disparityVisualization = clamp(disparity, 0.0, 1.0); // Adjust multiplier as needed
    outColor = vec4(vec3(disparityVisualization), 1.0);
}