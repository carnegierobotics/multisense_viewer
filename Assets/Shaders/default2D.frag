#version 450

layout (location = 0) in vec2 inUV;

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

float map(float value, float inMin, float inMax, float outMin, float outMax) {
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

    //float viewSpaceDepth = restoreDepth(depth);
    //// Convert view space Z to actual distance
    //float actualDistance = -viewSpaceDepth; // Since view space Z is typically negative going into the screen
    //float outVal = actualDistance;
    //outVal = map(outVal, 0.9, 1, 0, 1);

    vec3 depth = texture(samplerColorMap, inUV).rgb;  // Fetch from sample 0
    outColor = vec4(depth, 1.0f);
}