//
// Created by magnus on 2/21/22.
//

#include "CRLVirtualCamera.h"

void CRLVirtualCamera::initialize() {
    // INitialize camera

    // Initialize rendering
    // Get depth image size and point cloud size and create render data from this
    int vertexCount = 2048;
    int indexCount = 0;

    meshData = new MeshData();
    meshData->vertexCount = vertexCount;
    // Virtual class can generate some mesh data here
    meshData->vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));

}

void CRLVirtualCamera::start() {


}

void CRLVirtualCamera::stop() {

}

void CRLVirtualCamera::update() {

    uint32_t v = 0;
    for (uint32_t x = 0; x < 640; ++x){
        for (uint32_t z = 0; z < 480; ++z){
            v++;
        }
    }

}

CRLBaseCamera::MeshData* CRLVirtualCamera::getStream() {
    return  meshData;
}

CRLVirtualCamera::~CRLVirtualCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}
