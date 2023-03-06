#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;
layout(location = 3) in vec3 inCoords;
layout(location = 4) in vec2 imageDimmensions;


layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;

layout (set = 0, binding = 4) uniform colorConversionParams {
    mat4 intrinsic;
    mat4 extrinsic;
} mat;

layout(location = 0) out vec4 outColor;

void main()
{
    if (inCoords.z > 50 && inCoords.z < 0.1){
        outColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    // Project into color image
    vec4 colorCamCoords = 1/inCoords.z * mat.intrinsic * mat.extrinsic * vec4(inCoords, 1.0f);

    vec2 sampleCoords = vec2((colorCamCoords.x / imageDimmensions.x), colorCamCoords.y / imageDimmensions.y);

    // sample from color image
    vec3 tex = texture(samplerColorMap, sampleCoords).rgb;

    outColor = vec4(tex, 1.0f);
    //outColor = vec4(0.6f, 0.6f, 0.6f, 1.0f);

}