#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(location = 0) in vec3 Normal;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

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
    vec4 kernelFilters;
} info;

layout (set = 0, binding = 2) uniform sampler2D luma;

layout (set = 0, binding = 3) uniform sampler2D chromaU;

layout (set = 0, binding = 4) uniform sampler2D chromaV;


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

vec4 blurKernel(sampler2D textureSampler, vec2 texCoord) {
    int kernelSize = 2;
    vec2 texelSize = 1.0 / textureSize(textureSampler, 0);
    vec4 blurColor = vec4(0.0);

    for (int i = -kernelSize; i <= kernelSize; ++i) {
        for (int j = -kernelSize; j <= kernelSize; ++j) {
            vec2 offset = vec2(i, j) * texelSize;
            blurColor.rgb += texture(textureSampler, texCoord + offset).rgb;
        }
    }
    blurColor =  blurColor / float((2 * kernelSize + 1) * (2 * kernelSize + 1));
    return blurColor;// Only modify the red channel
}
vec4 edgeDetect(sampler2D textureSampler, vec2 texCoord) {
    // Texel size calculation
    vec2 texelSize = 1.0 / textureSize(textureSampler, 0);
    int kernelSize = 1;

    // Sobel kernel for edge detection in the x and y directions
    float kernelX[3][3];
    kernelX[0][0] = -1.0; kernelX[0][1] = 0.0; kernelX[0][2] = 1.0;
    kernelX[1][0] = -2.0; kernelX[1][1] = 0.0; kernelX[1][2] = 2.0;
    kernelX[2][0] = -1.0; kernelX[2][1] = 0.0; kernelX[2][2] = 1.0;

    float kernelY[3][3];
    kernelY[0][0] = -1.0; kernelY[0][1] = -2.0; kernelY[0][2] = -1.0;
    kernelY[1][0] = 0.0;  kernelY[1][1] = 0.0;  kernelY[1][2] = 0.0;
    kernelY[2][0] = 1.0;  kernelY[2][1] = 2.0;  kernelY[2][2] = 1.0;

    // Initialize edge colors for x and y directions for each channel
    vec3 edgeX = vec3(0.0);
    vec3 edgeY = vec3(0.0);

    // Apply the Sobel kernels
    for (int i = -kernelSize; i <= kernelSize; ++i) {
        for (int j = -kernelSize; j <= kernelSize; ++j) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec3 pixelValue = texture(textureSampler, texCoord + offset).rgb;

            edgeX += kernelX[i + 1][j + 1] * pixelValue;
            edgeY += kernelY[i + 1][j + 1] * pixelValue;
        }
    }

    // Calculate the magnitude of the edge for each channel
    vec3 edgeMagnitude = sqrt(edgeX * edgeX + edgeY * edgeY);

    // Return as a vec4
    return vec4(edgeMagnitude, 1.0);
}

vec4 emboss(sampler2D textureSampler, vec2 texCoord) {
    // Texel size calculation
    int kernelSize = 1;
    vec2 texelSize = 1.0 / textureSize(textureSampler, 0);

    // Embossing kernel
    float kernel[3][3];
    kernel[0][0] = -2.0; kernel[0][1] = -1.0; kernel[0][2] = 0.0;
    kernel[1][0] = -1.0; kernel[1][1] = 1.0;  kernel[1][2] = 1.0;
    kernel[2][0] = 0.0;  kernel[2][1] = 1.0;  kernel[2][2] = 2.0;
    // Initialize color
    vec3 embossedColor = vec3(0.0);

    // Apply the embossing kernel
    for (int i = -kernelSize; i <= kernelSize; ++i) {
        for (int j = -kernelSize; j <= kernelSize; ++j) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec3 pixelValue = texture(textureSampler, texCoord + offset).rgb;

            embossedColor += kernel[i + 1][j + 1] * pixelValue;
        }
    }

    // Add 0.5 to shift the color range from [-1, 1] to [0, 1]
    embossedColor = embossedColor + 0.5;

    // Return as a vec4
    return vec4(embossedColor, 1.0);
}
vec4 sharpening(sampler2D textureSampler, vec2 texCoord) {
    int kernelSize = 1;
    // Texel size calculation
    vec2 texelSize = 1.0 / textureSize(textureSampler, 0);

    // Sharpening kernel
    float kernel[3][3];
    kernel[0][0] = -1.0; kernel[0][1] = -1.0; kernel[0][2] = -1.0;
    kernel[1][0] = -1.0; kernel[1][1] = 9.0;  kernel[1][2] = -1.0;
    kernel[2][0] = -1.0; kernel[2][1] = -1.0; kernel[2][2] = -1.0;

    // Initialize color
    vec3 sharpenedColor = vec3(0.0);

    // Apply the sharpening kernel
    for (int i = -kernelSize; i <= kernelSize; ++i) {
        for (int j = -kernelSize; j <= kernelSize; ++j) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec3 pixelValue = texture(textureSampler, texCoord + offset).rgb;

            sharpenedColor += kernel[i + 1][j + 1] * pixelValue;
        }
    }

    // Clip the color values to be within [0, 1]
    sharpenedColor = clamp(sharpenedColor, 0.0, 1.0);

    // Return as a vec4
    return vec4(sharpenedColor, 1.0);
}


void main()
{
    vec2 zoom = vec2(info.zoom.x, info.zoom.y);

    float zoomCenterX = (info.zoom.w + 1.0f) * 0.5f;
    float zoomCenterY = (zoom.y + 1.0f) * 0.5f;

    float uvSampleX = (inUV.x - zoomCenterX) / info.zoom.z + zoomCenterX;
    float uvSampleY = (inUV.y - zoomCenterY) / info.zoom.z + zoomCenterY;

    vec2 uv = vec2(uvSampleX, uvSampleY);
    float r, g, b, y, u, v;
    mat3 colorMatrix = mat3(
                1,   1,       1,
                0,  -0.344,  1.72,
                1.402,   -0.71,   0);

    vec4 color;
    bool useInterpolation = info.normalize.w == 1.0f;
    if (useInterpolation){
        y = textureBicubic(luma, uv).r;
        u = textureBicubic(chromaU, uv).r - 0.5f;
        v = textureBicubic(chromaV, uv).r - 0.5f;
    } else {
        y = texture(luma, uv).r;
        u = texture(chromaU, uv).r - 0.5f;
        v = texture(chromaV, uv).r - 0.5f;
    }

    vec3 yuv = vec3(y, u, v);
    outColor = vec4(colorMatrix * yuv, 1.0f);

}