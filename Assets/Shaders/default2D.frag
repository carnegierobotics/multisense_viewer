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

layout (binding = 2) uniform sampler2DMS samplerColorMap;

layout (location = 0) out vec4 outColor;

double map(double value, double inMin, double inMax, double outMin, double outMax) {
    return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}



float restoreDepth(float z) {
    float near = 0.01; // Near clipping plane
    float far = 100.0; // Far clipping plane

    float A = -far / (far - near);
    float B = -far * near / (far -near);
    return (z - B) / A;
}


void main() {

    ivec2 texCoord = ivec2(inUV.x * 1280, inUV.y * 720);  // Transform normalized UVs to pixel coordinates
    vec4 depth = texelFetch(samplerColorMap, texCoord, 0);  // Fetch from sample 0

    float viewSpaceDepth = restoreDepth(depth.r);
    // Convert view space Z to actual distance
    double actualDistance = -viewSpaceDepth; // Since view space Z is typically negative going into the screen

    double outVal = actualDistance;
    outVal = map(outVal, 0.8, 1, 0, 1);
    outColor = vec4(vec3(outVal), 1.0f);
}