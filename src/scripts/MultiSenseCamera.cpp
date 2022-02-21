#include "MultiSenseCamera.h"

void MultiSenseCamera::setup() {

    this->vulkanDevice = renderUtils.device;
    // Create a 5x5 mesh


    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/triangle.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/triangle.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};
    renderUtils.shaders = shaders;

    virtualCamera = new CRLVirtualCamera();
    virtualCamera->initialize();


    CRLBaseCamera::MeshData *meshData = virtualCamera->getStream();
    transferDataStaging((MeshModel::Model::Vertex *) meshData->vertices, meshData->vertexCount, meshData->indices,
                        meshData->indexCount);
    MeshModel::createRenderPipeline(renderUtils, shaders);

}

void MultiSenseCamera::update() {

    virtualCamera->update();


}

void MultiSenseCamera::onUIUpdate(UISettings uiSettings) {
}


void MultiSenseCamera::generateGridPoints() {
    // 16*16 mesh as our ground
    // Get square size from input

    int xSize = 20;
    int zSize = 20;
    uint32_t vertexCount = xSize * zSize;
    auto *vertices = new MeshModel::Model::Vertex[vertexCount + 1];
    uint32_t indexCount = xSize * zSize * 6;
    auto *indices = new uint32_t[indexCount + 1];

    uint32_t v = 0;
    // Alloc memory for vertices and indices
    for (int z = 0; z < zSize; ++z) {
        for (int x = 0; x < xSize; ++x) {
            MeshModel::Model::Vertex vertex{};

            vertex.pos = glm::vec3((float) x, 0.0f, (float) z);
            vertices[v] = vertex;
            v++;
        }
    }


    // Normals
    int index = 0;
    for (int z = 0; z <= zSize - 2; ++z) {
        for (int x = 0; x <= xSize - 2; ++x) {
            glm::vec3 A = vertices[index].pos;
            glm::vec3 B = vertices[index + 1].pos;
            glm::vec3 C = vertices[index + 1 + xSize].pos;
            // Normals and stuff
            glm::vec3 AB = B - A;
            glm::vec3 AC = C - A;
            glm::vec3 normal = glm::cross(AC, AB);
            normal = glm::normalize(normal);
            // Give normal to last three vertices
            vertices[index].normal = normal;
            vertices[index + 1].normal = normal;
            vertices[index + 1 + xSize].normal = normal;

            index++;
        }
        index++;
    }

    int tris = 0;
    int vert = 0;
    for (int z = 0; z < zSize; ++z) {
        for (int x = 0; x < xSize; ++x) {
            // One quad
            indices[tris + 0] = vert;
            indices[tris + 1] = vert + 1;
            indices[tris + 2] = vert + xSize + 1;
            indices[tris + 3] = vert + 1;
            indices[tris + 4] = vert + xSize + 2;
            indices[tris + 5] = vert + xSize + 1;

            vert++;
            tris += 6;
        }
        vert++;
    }

    transferDataStaging(vertices, vertexCount, indices, indexCount);

    delete[] vertices;
    delete[] indices;
}

void MultiSenseCamera::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    MeshModel::draw(commandBuffer, i);
}
