#version 450

// Interpolated from vertex
layout (location = 0) in vec2 vUV;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vWorldPos;
layout (location = 3) flat in uint vInstIdx;

// Same push-constant block
layout (push_constant) uniform BatchPC {
    uint transformBase;
    uint materialBase;
} pc;

// Global UBO (set 0 / binding 0)
layout (set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
    float numLights;
    vec4 lightPos[16];
} uGlobal;

// GLSL: match std430 layout
struct MaterialBufferObject {
    vec4 baseColor;      // matches glm::vec4
    float specular;      // matches float
    float diffuse;
    float phongExponent;// bool→float: 1.0=useTexture, 0.0=use baseColor
    float useVertexColor;// bool→float: 1.0=useTexture, 0.0=use baseColor
    vec4 emissiveFactor; // matches glm::vec4
};



// Material SSBO (set 2 / binding 0)
layout (set = 2, binding = 0) readonly buffer MaterialBuf {
    MaterialBufferObject mat[];
} uMaterials;

// Diffuse texture (set 3 / binding 0)
layout (set = 3, binding = 0) uniform sampler2D uTexture;

layout (location = 0) out vec4 outColor;

void main() {
    /*
    // compute material index
    uint mIdx = pc.materialBase + vInstIdx;
    MaterialBufferObject mbo = uMaterials.mat[mIdx];

    // fetch albedo
    vec3 texCol = texture(uTexture, vUV).rgb;
    vec3 albedo = (mbo.useVertexColor > 0.5)
    ? texCol * mbo.baseColor.rgb
    : mbo.baseColor.rgb;

    // lighting
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uGlobal.cameraPos - vWorldPos);
    vec3 color = 0.05 * albedo; // ambient

    // loop over lights
    for (uint i = 0; i < uint(uGlobal.numLights); ++i) {
        vec3 L = normalize(uGlobal.lightPos[i].xyz - vWorldPos);
        float diff = max(dot(N, L), 0.0);
        color += diff * albedo;
        // Blinn-Phong spec
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), mbo.specular);
        color += spec * vec3(mbo.specular);
    }

    // emissive
    color += mbo.emissiveFactor.rgb;

    outColor = vec4(color, 1.0);
    */

    uint mIdx = pc.materialBase + vInstIdx;
    MaterialBufferObject mbo = uMaterials.mat[mIdx];

    vec3 normal = normalize(vNormal);
    vec3 vertPos = vWorldPos.xyz;
    vec3 lightPos = uGlobal.lightPos[0].xyz;
    vec3 lightDir = normalize(lightPos - vertPos);

    vec3 viewDir    = normalize(uGlobal.cameraPos - vertPos);

    vec3 halfwayDir = normalize(lightDir + viewDir);


    vec3 texCol = texture(uTexture, vUV).rgb;
    vec3 albedo = (mbo.useVertexColor > 0.5)
    ? texCol * mbo.baseColor.rgb
    : mbo.baseColor.rgb;

    vec3 ambient = 0.2f * albedo;

    float diff = max(dot(lightDir, normal), 0.0) * mbo.diffuse;
    vec3 diffuse = diff * albedo;

    float spec = pow(max(dot(normal, halfwayDir), 0.0), mbo.phongExponent) * mbo.specular;
    vec3 specular = spec * albedo;

    // use the gamma corrected color in the fragment
    outColor =vec4(ambient + diffuse + specular + mbo.emissiveFactor.x, mbo.baseColor.a);
}
