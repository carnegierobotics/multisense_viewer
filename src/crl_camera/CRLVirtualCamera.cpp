//
// Created by magnus on 2/21/22.
//

#include "CRLVirtualCamera.h"

void CRLVirtualCamera::connect(CRLCameraDataType source) {
    //CRLBaseCamera::connect(VIRTUAL_CAMERA);

}

void CRLVirtualCamera::start(std::string string, std::string dataSourceStr) {


}

void CRLVirtualCamera::stop(std::string dataSourceStr) {

}

void CRLVirtualCamera::update() {


    //auto *vP = (MeshModel::Model::Vertex *) meshData->vertices;
    //auto y = (float) sin(glm::radians(render.runTime * 60.0f));

    for (int i = 0; i < 720; ++i) {
        //vP[point].pos.y = y;
        point++;
        if (point >= meshData->vertexCount)
            point = 0;
    }
}

CRLBaseCamera::PointCloudData *CRLVirtualCamera::getStream() {
    return meshData;
}

void  CRLVirtualCamera::getVirtualCameraMetaData() {

    // Just populating it with some hardcoded data :)
    // - DevInfo
    cameraInfo.devInfo.name = "CRL Virtual Camera";
    cameraInfo.devInfo.imagerName = "Virtual";
    cameraInfo.devInfo.serialNumber = "25.8069758011"; // Root of all evil
    // - getImageCalibration
    cameraInfo.netConfig.ipv4Address = "Knock knock";

}

CRLVirtualCamera::~CRLVirtualCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}

