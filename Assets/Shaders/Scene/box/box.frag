#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 fragPos;
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


layout (location = 0) out vec4 outFragColor;

void main()
{
    float ambientStrength = 0.9;
    vec3 ambient = ambientStrength * info.lightColor.xyz;
    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(info.lightPos.xyz - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * info.lightColor.xyz;

    // specular
    float specularStrength = 0.85;
    vec3 viewDir = normalize(info.viewPos.xyz - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec3 specular = specularStrength * spec * info.lightColor.xyz;

    vec3 result = (ambient + diffuse + specular) * info.objectColor.xyz;
    outFragColor = vec4(result, 1.0f);

}