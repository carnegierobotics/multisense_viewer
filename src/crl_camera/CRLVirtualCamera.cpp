//
// Created by magnus on 2/21/22.
//

#include "CRLVirtualCamera.h"

void CRLVirtualCamera::initialize() {
    // INitialize camera

    // Initialize rendering
    // Get depth image size and point cloud size and create render data from this
    int pcW = 1024,  pcH = 544;
    int vertexCount = pcW * pcH;
    int indexCount = 0;

    meshData = new MeshData();
    meshData->vertexCount = vertexCount;
    // Virtual class can generate some mesh data here
    meshData->vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));

    uint32_t v = 0;
    auto *vP = (MeshModel::Model::Vertex *) meshData->vertices;
    for (uint32_t x = 0; x < pcW; ++x) {
        for (uint32_t z = 0; z < pcH; ++z) {
            MeshModel::Model::Vertex vertex{};
            vertex.pos = glm::vec4((float) x / 100, 0.0f, (float) z / 100, 1.0f);
            vP[v] = vertex;
            v++;
        }
    }

}

void CRLVirtualCamera::start() {


}

void CRLVirtualCamera::stop() {

}

void CRLVirtualCamera::update(Base::Render render) {

    auto *vP = (MeshModel::Model::Vertex *) meshData->vertices;
    auto y = (float) sin(glm::radians(render.runTime * 60.0f));

    for (int i = 0; i < 1000; ++i) {
        vP[point].pos.y = y;
        point++;


        if (point>= meshData->vertexCount)
            point = 0;
    }


}

CRLBaseCamera::MeshData *CRLVirtualCamera::getStream() {
    return meshData;
}

CRLVirtualCamera::~CRLVirtualCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}
