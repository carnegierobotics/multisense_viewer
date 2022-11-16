#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform Colors {
    vec4 objectColor;
    vec4 lightColor;
    vec4 lightPos;
    vec4 viewPos;
} colors;

layout (set = 0, binding = 2) uniform sampler2D luma;


layout (set = 0, binding = 3) uniform sampler2D chromaU;

layout (set = 0, binding = 4) uniform sampler2D chromaV;


void main()
{

    float r, g, b, y, u, v;
    mat3 colorMatrix = mat3(
                1,   0,       1.402,
                1,  -0.344,  -0.714,
                1,   1.772,   0);


    y = texture(luma, inUV).r;
    u = texture(chromaU, vec2(inUV.x, inUV.y)).r - 0.5f;
    v = texture(chromaV, vec2(inUV.x, inUV.y)).r - 0.5f;

    vec3 yuv = vec3(y, u, v);
    outColor = vec4(yuv*colorMatrix, 1.0f);

    // Possibly expand range here if using TV YUV range and not PC YUV range.
    //yuv = rescale_yuv(yuv);
    
    //r = y + (1.402*v);
    //g = y - (0.344136*u) - (0.714136*v);
    //b = y + (1.772*u);


    //outColor =  vec4(r, g, b, 1.0);

}