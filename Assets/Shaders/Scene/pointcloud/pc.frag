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

float Triangular( float f )
{
    f = f / 2.0;
    if( f < 0.0 )
    {
        return ( f + 1.0 );
    }
    else
    {
        return ( 1.0 - f );
    }
    return 0.0;
}

// Function to get interpolated texel data from a texture with GL_NEAREST property.
// Bi-Linear interpolation is implemented in this function with the
// help of nearest four data.
vec4 BiCubic( sampler2D textureSampler, vec2 TexCoord )
{
    float texelSizeX = 1.0 / imageDimmensions.x; //size of one texel
    float texelSizeY = 1.0 / imageDimmensions.y; //size of one texel
    vec4 nSum = vec4( 0.0, 0.0, 0.0, 0.0 );
    vec4 nDenom = vec4( 0.0, 0.0, 0.0, 0.0 );
    float a = fract( TexCoord.x * imageDimmensions.x ); // get the decimal part
    float b = fract( TexCoord.y * imageDimmensions.y ); // get the decimal part
    for( int m = -1; m <=2; m++ )
    {
        for( int n =-1; n<= 2; n++)
        {
            vec4 vecData = texture(textureSampler,
            TexCoord + vec2(texelSizeX * float( m ),
            texelSizeY * float( n )));
            float f  = Triangular( float( m ) - a );
            vec4 vecCooef1 = vec4( f,f,f,f );
            float f1 = Triangular ( -( float( n ) - b ) );
            vec4 vecCoeef2 = vec4( f1, f1, f1, f1 );
            nSum = nSum + ( vecData * vecCoeef2 * vecCooef1  );
            nDenom = nDenom + (( vecCoeef2 * vecCooef1 ));
        }
    }
    return nSum / nDenom;
}

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
    outColor = BiCubic(samplerColorMap, sampleCoords);
    //outColor = vec4(0.6f, 0.6f, 0.6f, 1.0f);

}