#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 fragPos;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inFragPos; // World space position
layout (location = 4) in vec4 vertexColor; // World space position


layout (binding = 0) uniform CameraUBO
{
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

layout (set = 1, binding = 0) uniform Info {
    vec4 baseColor;
    float metallic;
    float roughness;
    float isDisparity;
    vec4 emissiveFactor;
} info;

layout (set = 1, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

vec3 calculatePhongLighting(vec3 normal, vec3 fragPos, vec3 viewDir, vec3 lightDir) {
    // Phong lighting components
    vec3 ambientColor = vec3(0.2, 0.2, 0.2); // Ambient color
    vec3 diffuseColor = vec3(1.0, 1.0, 1.0); // Diffuse color
    vec3 specularColor = vec3(1.0, 1.0, 1.0); // Specular color
    float shininess = 32.0;

    // Ambient
    vec3 ambient = ambientColor;

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    // Combine results
    return (ambient + diffuse + specular);
}

void main()
{

    vec3 lightDir = normalize(vec3(camera.position)); // Fixed sunlight direction

    vec3 norm = normalize(inNormal);
    vec3 viewDir = normalize(camera.position - inFragPos);

    vec3 phongLighting = calculatePhongLighting(norm, inFragPos, viewDir, lightDir);

    vec3 texColor = texture(samplerColorMap, inUV).rgb;
   /*
    if (info.isDisparity == 1){
        float color = texColor.r * 64;
        texColor = vec4(vec3(color), 1.0);
    }
    */
    outColor = vec4(phongLighting, 1.0) * info.baseColor * vec4(texColor, 1.0);
    //outColor = vec4(phongLighting, 1.0) * vertexColor ;
    //outColor = texColor * info.baseColor ;
}
