//
// Created by magnus on 3/1/22.
//

#include "CRLMultiSenseS30.h"


void CRLMultiSenseS30::initialize() {
    // Initialize camera

    // Initialize rendering
    // Get depth image size and point cloud size and create render data from this
    int pcW = 1024,  pcH = 544;
    int vertexCount = pcW * pcH;
    meshData = new PointCloudData();
    meshData->vertexCount = vertexCount;
    // Virtual class can generate some mesh data here
    meshData->vertices = calloc(vertexCount, sizeof(MeshModel::Model::Vertex));


}

void CRLMultiSenseS30::start() {


}

void CRLMultiSenseS30::stop() {

}

void CRLMultiSenseS30::update(Base::Render render) {


}

CRLBaseCamera::PointCloudData *CRLMultiSenseS30::getStream() {
    return meshData;
}

CRLMultiSenseS30::~CRLMultiSenseS30() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}
