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
    bool useVertexColor;
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

    vec3 normal = normalize(inNormal);
    vec3 vertPos = fragPos.xyz;
    vec3 lightPos = info.lightPosition[0].xyz;
    vec3 lightDir = normalize(lightPos - vertPos);

    vec3 viewDir    = normalize(camera.position - vertPos);

    vec3 halfwayDir = normalize(lightDir + viewDir);

    float spec = pow(max(dot(normal, halfwayDir), 0.0), 128);
    vec3 specular = vec3(0.3) * spec;

    vec3 ambient = 0.05 * info.baseColor.xyz;

    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * info.baseColor.xyz;

    // use the gamma corrected color in the fragment
    outColor =vec4(ambient + diffuse + specular + info.emissiveFactor.x, 1.0);

}
