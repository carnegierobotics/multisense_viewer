#version 450

layout (binding = 0) uniform CameraUBO {
    mat4 projection;
    mat4 view;
    vec3 pos;   // not used below, but that’s fine
} camera;

layout (binding = 1) uniform ModelUBO {
    mat4 model; // Typically used if you had a single model transform
} ubo;

struct VertexData {
    vec3 position;
    vec3 normal;
    vec2 uv0;
    vec2 uv1;
    vec4 color;
};

layout(std430, set=2, binding=0) readonly buffer CameraGizmoSSBO {
    VertexData vertices[];
} vertexData;

layout(std430, set=2, binding=1) readonly buffer IndexBufferSSBO {
    uint indices[];
} indexBuffer;


layout(location = 0) out vec4 outColor;

void main()
{
    // Fetch the index for this vertex
    int idx = int(indexBuffer.indices[gl_VertexIndex]);

    // Now also fetch the per‐instance data:
    // gl_InstanceIndex gives us which instance we are currently drawing.

    vec3 position = vertexData.vertices[idx].position;
    vec4 color    = vertexData.vertices[idx].color;

    // Combine with (optional) "ubo.model".
    // For instance, you might combine them if you have a *global*
    // transform from the ModelUBO, or you might skip ubo.model entirely
    // and rely purely on the instance model.

    // Transform to world -> clip space
    vec4 worldPos = ubo.model * vec4(position, 1.0);
    gl_Position = camera.projection * camera.view * worldPos;

    outColor = color;
}
