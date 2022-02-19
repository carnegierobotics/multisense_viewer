#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 FragPos;
layout (location = 4) in mat4 model;

layout(set = 0, binding = 1) uniform INFO {
    vec4 lightColor;
    vec4 objectColor;
    vec4 lightPos;
    vec4 viewPos;

} info;

layout(set = 0, binding = 2) uniform SELECT {
    float map;
} select ;


layout (set = 0, binding = 3) uniform sampler2D samplerColorMap;
layout (set = 0, binding = 4) uniform sampler2D samplerTextureMap;

layout (location = 0) out vec4 outFragColor;


vec3 getNormal()
{
    // Perturb normal, see http://www.thetenthplanet.de/archives/1180
    vec3 tangentNormal = texture(samplerTextureMap, inUV).xyz * 2.0 - 1.0;

    vec3 q1 = dFdx(inPos);
    vec3 q2 = dFdy(inPos);
    vec2 st1 = dFdx(inUV);
    vec2 st2 = dFdy(inUV);

    vec3 N = normalize(inNormal);
    vec3 T = normalize(q1 * st2.t - q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

void main()
{
    vec3 color;
    vec3 normals = inNormal;
    if (select.map == 0){
        color = vec3(0.7, 0.7, 0.7);
    }
    mat3 TBN = mat3(transpose(inverse(model)));
    if (select.map == 1){
        color = texture(samplerColorMap, inUV).rgb;
    }
    if (select.map == 2){
        color = texture(samplerColorMap, inUV).rgb;

        normals = getNormal();

    }

    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * info.lightColor.rgb;

    // diffuse
    vec3 norm = normalize(normals);
    vec3 lightDir = normalize((info.lightPos.xyz) - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * info.lightColor.rgb;

    // specular
    float specularStrength = 0.7;
    vec3 viewDir = normalize(info.viewPos.xyz - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec3 specular = specularStrength * spec * info.lightColor.rgb;

    vec3 result = (ambient + diffuse + specular) * color;

    outFragColor = vec4(result, 1.0);


}