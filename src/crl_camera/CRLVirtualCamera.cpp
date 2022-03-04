//
// Created by magnus on 2/21/22.
//

#include "CRLVirtualCamera.h"

void CRLVirtualCamera::initialize(CRLCameraDataType source) {
    // INitialize camera

    // Initialize rendering
    // Get depth image size and point cloud size and create render data from this
    if (source == CrlPointCloud) {
        // Load sample video
        int pcW =1280, pcH = 720;
        int vertexCount = pcW * pcH;
        int indexCount = 0;

        meshData = new PointCloudData();
        meshData->vertexCount = vertexCount;
        // Virtual class can generate some mesh data here
        meshData->vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));

        uint32_t v = 0;
        auto *vP = (MeshModel::Model::Vertex *) meshData->vertices;
        for (uint32_t x = 0; x < pcW; ++x) {
            for (uint32_t z = 0; z < pcH; ++z) {
                MeshModel::Model::Vertex vertex{};
                vertex.pos = glm::vec3((float) x / 50, 0.0f, (float) z / 50);
                vertex.uv0 = glm::vec2((float) x / (float) pcW, (float) z / (float) pcH);
                vP[v] = vertex;
                v++;
            }
        }
    } else if (source == CrlImage) {
        // Load sample video
        int vertexCount = 4;
        int indexCount = 2 * 3;
        imageData = new ImageData();
        imageData->quad.vertexCount = vertexCount;
        imageData->quad.indexCount = indexCount;
        // Virtual class can generate some mesh data here
        imageData->quad.vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));
        imageData->quad.indices = static_cast<uint32_t *>(calloc(indexCount, sizeof(uint32_t)));

        auto *vP = (MeshModel::Model::Vertex *) imageData->quad.vertices;
        auto *iP = (uint32_t *) imageData->quad.indices;

        MeshModel::Model::Vertex vertex[4];
        vertex[0].pos = glm::vec3(0.0f, 0.0f, 0.0f);
        vertex[1].pos = glm::vec3(1.0f, 0.0f, 0.0f);
        vertex[2].pos = glm::vec3(0.0f, 0.0f, 1.0f);
        vertex[3].pos = glm::vec3(1.0f, 0.0f, 1.0f);

        vertex[0].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        vertex[1].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        vertex[2].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        vertex[3].normal = glm::vec3(0.0f, 1.0f, 0.0f);

        vertex[0].uv0 = glm::vec2(0.0f, 0.0f);
        vertex[1].uv0 = glm::vec2(1.0f, 0.0f);
        vertex[2].uv0 = glm::vec2(0.0f, 1.0f);
        vertex[3].uv0 = glm::vec2(1.0f, 1.0f);
        vP[0] = vertex[0];
        vP[1] = vertex[1];
        vP[2] = vertex[2];
        vP[3] = vertex[3];
        // indices
        iP[0] = 0;
        iP[1] = 1;
        iP[2] = 2;
        iP[3] = 1;
        iP[4] = 2;
        iP[5] = 3;
    }


}

void CRLVirtualCamera::start() {


}

void CRLVirtualCamera::stop() {

}

void CRLVirtualCamera::update(Base::Render render) {


    auto *vP = (MeshModel::Model::Vertex *) meshData->vertices;
    auto y = (float) sin(glm::radians(render.runTime * 60.0f));

    for (int i = 0; i < 720; ++i) {
        vP[point].pos.y = y;
        point++;


        if (point >= meshData->vertexCount)
            point = 0;
    }
}

CRLBaseCamera::PointCloudData *CRLVirtualCamera::getStream() {
    return meshData;
}

CRLVirtualCamera::~CRLVirtualCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}

CRLBaseCamera::ImageData *CRLVirtualCamera::getImageData() {
    return imageData;
}
