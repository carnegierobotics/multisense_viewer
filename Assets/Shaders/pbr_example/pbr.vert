#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV0;


layout (set = 0, binding = 0) uniform UBO
{
    mat4 projection;
    mat4 view;
    mat4 model;
    vec3 camPos;
} ubo;

#define MAX_NUM_JOINTS 128

layout (set = 2, binding = 0) uniform UBONode {
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
    float jointCount;
} node;

layout (location = 0) out vec3 outWorldPos;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV0;
layout (location = 3) out vec2 outUV1;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    vec4 locPos;
    if (node.jointCount > 0.0) {
        // Mesh is skinned
        //mat4 skinMat =
        //inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
        //inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
        //inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
        //inWeight0.w * node.jointMatrix[int(inJoint0.w)];

        //locPos = ubo.model * node.matrix * skinMat * vec4(inPos, 1.0);
        //outNormal = normalize(transpose(inverse(mat3(ubo.model * node.matrix * skinMat))) * inNormal);
    } else {
        locPos = ubo.model * node.matrix * vec4(inPos, 1.0);
        outNormal = normalize(transpose(inverse(mat3(ubo.model * node.matrix))) * inNormal);
    }
    locPos.y = -locPos.y;
    outWorldPos = locPos.xyz / locPos.w;
    outUV0 = inUV0;
    //outUV1 = inUV1;
    gl_Position =  ubo.projection * ubo.view * vec4(outWorldPos, 1.0);
}