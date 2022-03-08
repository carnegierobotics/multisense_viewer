//
// Created by magnus on 2/21/22.
//

#include "CRLVirtualCamera.h"

void CRLVirtualCamera::connect(CRLCameraDataType source) {
    CRLBaseCamera::connect(VIRTUAL_CAMERA);

}

void CRLVirtualCamera::start(std::string string) {


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

CRLBaseCamera::ImageData *CRLVirtualCamera::getImageData() const {
    return imageData;
}
