#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout (push_constant) uniform Material {
    vec2 pos;
} material;


layout(binding = 1, set = 0) uniform Info {
    vec4 lightDir;
    vec4 zoom;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float debugViewInputs;
    float lod;
    vec2 pad;
} info;

layout (set = 0, binding = 2) uniform sampler2D samplerColorMap;

// https://stackoverflow.com/questions/13501081/efficient-bicubic-filtering-code-in-glsl
vec4 cubic(float x)
{
    float x2 = x * x;
    float x3 = x2 * x;
    vec4 w;
    w.x =   -x3 + 3*x2 - 3*x + 1;
    w.y =  3*x3 - 6*x2       + 4;
    w.z = -3*x3 + 3*x2 + 3*x + 1;
    w.w =  x3;
    return w / 6.f;
}

vec4 textureBicubic(sampler2D samplerMap, vec2 texCoords){

    vec2 texSize = textureSize(samplerMap, 0);
    vec2 invTexSize = 1.0 / texSize;
    texCoords = texCoords * texSize - 0.5;
    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture(samplerMap, offset.xz);
    vec4 sample1 = texture(samplerMap, offset.yz);
    vec4 sample2 = texture(samplerMap, offset.xw);
    vec4 sample3 = texture(samplerMap, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(
    mix(sample3, sample2, sx), mix(sample1, sample0, sx)
    , sy);
}

void main()
{

    float val = info.zoom.z;
    vec2 zoom = vec2(info.zoom.x, info.zoom.y);

    float uvSampleX = (inUV.x - info.zoom.w) / info.zoom.z + info.zoom.w;
    float uvSampleY = (inUV.y - zoom.y) / info.zoom.z + zoom.y;

    vec4 color;
    bool useInterpolation = info.zoom.x == 1.0f;
    if (useInterpolation){
        color = textureBicubic(samplerColorMap, vec2(uvSampleX, uvSampleY));
    } else {
        color = texture(samplerColorMap, vec2(uvSampleX, uvSampleY));
    }
    outColor = vec4(color.r, color.r, color.r, 1.0);

}