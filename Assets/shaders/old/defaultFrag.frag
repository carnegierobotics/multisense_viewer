#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 normals;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 Normal;

layout(binding = 1) uniform sampler2D texSampler;
layout(location = 0) out vec4 outColor;

void main() {

    vec3 color = texture(texSampler, fragTexCoord).rgb;
    float avg = (color.r + color.g + color.b) / 3.0;
    color.rgb = vec3(avg);

    outColor = vec4(color, 1);
}