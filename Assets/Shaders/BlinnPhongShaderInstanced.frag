#version 450

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 fragPos;
layout (location = 2) in vec3 inNormal;

layout (binding = 0) uniform CameraUBO {
    mat4 projection;
    mat4 view;
    vec3 position; // renamed to match fragment shader
} camera;

layout (set = 1, binding = 0) uniform Info {
    vec4 baseColor;
    float specular;
    float diffuse;
    vec2 _pad0;        // Ensure 16-byte alignment
    vec4 emissiveFactor;
    float numLightSources;
    vec4 lightPosition[32]; // Expanded vec3 -> vec4 for alignment
    vec4 lightNormal[32];   // Expanded vec3 -> vec4 for alignment
    bool useTexture;
} info;

layout (set = 1, binding = 1) uniform sampler2D samplerColorMap;

layout (location = 0) out vec4 outColor;

// Calculates diffuse and specular contributions for one light (ambient is added separately)
vec3 calculatePhongLighting(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 specularColor, float shininess) {
    // Diffuse term
    float diff = max(dot(-normal, lightDir), 0.0);
    vec3 diffuseComponent = diff * vec3(1.0); // white light assumed

    // Specular term
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specularComponent = spec * specularColor;

    return diffuseComponent + specularComponent;
}

void main()
{
    // 1. sample texture
    vec3 texColor = texture(samplerColorMap, inUV).rgb;

    // 2. combine with your uniform baseColor
    vec3 albedo;

    if (info.useTexture){
        albedo =  texColor * info.baseColor.xyz;
    } else {
        albedo = info.baseColor.xyz;
    }

    // 3. recompute diffuse + ambient using albedo
    vec3 normal   = normalize(inNormal);
    vec3 vertPos  = fragPos.xyz;
    vec3 lightPos = info.lightPosition[0].xyz;
    vec3 lightDir = normalize(lightPos - vertPos);
    vec3 viewDir  = normalize(camera.position - vertPos);

    // ambient
    vec3 ambient  = 0.05 * albedo;

    // diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo;

    // Blinn-Phong specular (unchanged)
    vec3 halfway = normalize(lightDir + viewDir);
    float specPow = pow(max(dot(normal, halfway), 0.0), 128.0);
    vec3 specular = specPow * vec3(info.specular);

    // emissive
    vec3 emissive = info.emissiveFactor.xyz;

    // final color
    vec3 color = ambient + diffuse + specular + emissive;
    outColor = vec4(color, 1.0);
}