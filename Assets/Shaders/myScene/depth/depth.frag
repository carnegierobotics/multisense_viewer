#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec2 mousePos;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 0) uniform Colors {
    vec4 objectColor;
    vec4 lightColor;
    vec4 lightPos;
    vec4 viewPos;
} colors;

layout(set = 0, binding = 2) uniform SELECT {
    float map;
} select ;

layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;

layout(set = 0, binding = 4) uniform ZOOM {
    float zoom;

} zoom;


void main()
{
    vec2 uv = inUV * zoom.zoom;
    vec3 tex = texture(samplerColorMap, uv).rgb * 65535 / 4096 * 2; // magic number that makes it work
    vec3 color = tex;

    outColor = vec4(color.r, color.r, color.r, 1.0);
}