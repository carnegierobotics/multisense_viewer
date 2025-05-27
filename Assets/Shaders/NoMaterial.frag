#version 450

// Interpolated from vertex
layout (location = 0) in vec2 vUV;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vWorldPos;
layout (location = 3) flat in uint vInstIdx;
layout (location = 4) in vec4 vertexColor;

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
    uint mIdx = pc.materialBase + vInstIdx;
    MaterialBufferObject mbo = uMaterials.mat[mIdx];

    // use the gamma corrected color in the fragment
    outColor = vec4(mbo.baseColor);
    if (mbo.useVertexColor < 0.5){
        outColor = vertexColor;
    }
}
