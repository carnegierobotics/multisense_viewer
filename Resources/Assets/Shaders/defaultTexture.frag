#version 450

layout(location = 0) in vec4 inColor;

layout (binding = 0) uniform CameraUBO
{
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (set = 1, binding = 0) uniform Info {
    vec4 baseColor;
    float specular;
    float diffuse;
    vec2 _pad0;        // Ensure 16-byte alignment
    vec4 emissiveFactor;
    float numLightSources;
    vec4 lightPosition[32]; // Expanded vec3 -> vec4 for alignment
    vec4 lightNormal[32];   // Expanded vec3 -> vec4 for alignment
    bool useVertexColor;
} info;

layout (set = 1, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;


void main()
{
    outColor =  info.baseColor;

    if (info.useVertexColor)
            outColor = inColor;
}
