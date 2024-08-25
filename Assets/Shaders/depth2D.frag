#version 450

layout (location = 0) in vec2 inUV;
layout (binding = 0) uniform sampler2D depthColorMap;
layout (location = 0) out vec4 outColor;

layout (binding = 1) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
    vec3 camPos;
} ubo;

vec3 jetColorMap(float value) {
    vec3 color;
    float r = clamp(1.5 - abs(4.0 * value - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * value - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * value - 1.0), 0.0, 1.0);
    color = vec3(b, g, r);
    return color;
}

void main() {
    // Sample the depth value from the depth map
    float depth = texture(depthColorMap, inUV).r;

    // Reconstruct the clip space position
    vec4 clipSpacePos = vec4(0.0, 0.0, depth * 2.0 - 1.0, 1.0); // NDC range in Vulkan is [0, 1] -> Clip space range is [-1, 1]

    // Transform back to world space
    vec4 worldPos = inverse(ubo.view) * inverse(ubo.projection) * clipSpacePos;
    worldPos /= worldPos.w; // Perspective divide

    // Extract the world space depth
    float worldDepth = worldPos.z;

    float maxDisparity = 700.0;
    float baseline = 0.30;

    float disparity = (-baseline * 1000) / worldDepth;
    //float disparity = maxDisparity / (worldDepth + baseline);
    // Normalize disparity
    float normalizedDisparity = disparity / maxDisparity;

    // Apply jet colormap
    vec3 color = jetColorMap(normalizedDisparity);

    // Output the color
    outColor = vec4(vec3(normalizedDisparity), 1.0);
}