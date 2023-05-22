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
    vec4 zoomTranslate;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float debugViewInputs;
    float lod;
    vec2 pad;
    vec4 normalize;
} info;

layout (set = 0, binding = 2) uniform sampler2D samplerColorMap;


// https://stackoverflow.com/questions/13501081/efficient-bicubic-filtering-code-in-glsl
vec4 cubic(float v)
{
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w);
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

const mat3 sobelX = mat3(
vec3(-1, 0, 1),
vec3(-2, 0, 2),
vec3(-1, 0, 1)
);

const mat3 sobelY = mat3(
vec3(-1, -2, -1),
vec3(0, 0, 0),
vec3(1, 2, 1)
);


vec2 sobel3x3(sampler2D tex, vec2 uv, vec2 texSize) {
    vec2 texelSize = 1.0 / texSize;
    vec2 gradient = vec2(0.0);

    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            vec2 offset = vec2(x - 1, y - 1);
            vec2 texelUV = uv + offset * texelSize;
            float grayscale = texture(tex, texelUV).r;
            gradient.x += grayscale * sobelX[y][x];
            gradient.y += grayscale * sobelY[y][x];
        }
    }

    return gradient;
}


const int octaves = 8;
const float lacunarity = 2.0;
const float persistence = 0.5;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fractalNoise(vec2 st) {
    float amplitude = 1.0;
    float frequency = 1.0;
    float noiseValue = 0.0;

    for (int i = 0; i < octaves; ++i) {
        noiseValue += amplitude * noise(st * frequency);
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return noiseValue;
}

const int kernelSize = 1;
const float blurAmount = 0.3; // Adjust the blur intensity

vec4 blurKernel(sampler2D textureSampler, vec2 texCoord) {
    vec2 texelSize = 1.0 / textureSize(textureSampler, 0);
    vec4 blurColor = vec4(0.0);

    for (int i = -kernelSize; i <= kernelSize; ++i) {
        for (int j = -kernelSize; j <= kernelSize; ++j) {
            vec2 offset = vec2(i, j) * texelSize;
            blurColor.r += texture(textureSampler, texCoord + offset).r;
        }
    }

    return vec4(blurColor.r / float((2 * kernelSize + 1) * (2 * kernelSize + 1)),
    0.0, 0.0, 1.0); // Only modify the red channel
}

void main()
{
    float scaleFactor = 4.0; // Adjust this value to control the smoothness of the bicubic sampling
    float val = info.zoom.z;
    vec2 zoom = vec2(info.zoom.x, info.zoom.y);

    float zoomCenterX = (info.zoom.w + 1.0f) * 0.5f;
    float zoomCenterY = (zoom.y + 1.0f) * 0.5f;

    float uvSampleX = (inUV.x - zoomCenterX) / info.zoom.z + zoomCenterX;
    float uvSampleY = (inUV.y - zoomCenterY) / info.zoom.z + zoomCenterY;


    vec4 color;
    bool useInterpolation = info.normalize.w == 1.0f;

    vec2 gradient = sobel3x3(samplerColorMap, vec2(uvSampleX, uvSampleY), textureSize(samplerColorMap, 0));
    float gradientMagnitude = length(gradient) / 4.0;


    if (useInterpolation){
        color = textureBicubic(samplerColorMap, vec2(uvSampleX, uvSampleY));

        //vec4 sobelColor = vec4(vec3(gradientMagnitude), 1.0);
        //color = mix(texture(samplerColorMap, vec2(uvSampleX, uvSampleY)), sobelColor, 0.7);
        //color = blurKernel(samplerColorMap, vec2(uvSampleX, uvSampleY));

    } else {
        color = texture(samplerColorMap, vec2(uvSampleX, uvSampleY));
    }
    outColor = vec4(color.r, color.r, color.r, 1.0);

}