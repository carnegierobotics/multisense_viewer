//
// Created by magnus on 3/1/22.
//

#include "CRLPhysicalCamera.h"


void CRLPhysicalCamera::connect() {
    online = CRLBaseCamera::connect(DEFAULT_CAMERA_IP);

}

void CRLPhysicalCamera::start(std::string string) {

    crl::multisense::DataSource source = crl::multisense::Source_Disparity;
    bool status = cameraInterface->startStreams(crl::multisense::Source_Disparity);
    printf("Started stream %d\n", status);
}

void CRLPhysicalCamera::stop() {

}

void CRLPhysicalCamera::update(Base::Render render, crl::multisense::image::Header *pHeader) {

}

CRLBaseCamera::PointCloudData *CRLPhysicalCamera::getStream() {
    return meshData;
}

crl::multisense::image::Header CRLPhysicalCamera::getImage(){
    return imageP;
}

CRLPhysicalCamera::~CRLPhysicalCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}

CRLBaseCamera::ImageData *CRLPhysicalCamera::getImageData() {

    return imageData;
}

CRLBaseCamera::CameraInfo CRLPhysicalCamera::getInfo() {
    return CRLBaseCamera::cameraInfo;
}
