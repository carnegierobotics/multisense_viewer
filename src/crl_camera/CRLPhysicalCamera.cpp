//
// Created by magnus on 3/1/22.
//

#include "CRLPhysicalCamera.h"


void CRLPhysicalCamera::connect() {
    online = CRLBaseCamera::connect(DEFAULT_CAMERA_IP);

}

void CRLPhysicalCamera::start(std::string string) {

    crl::multisense::DataSource source = crl::multisense::Source_Disparity;
    // Set mode first
    std::string delimiter = "x";

    size_t pos = 0;
    std::string token;
    std::vector<uint32_t> widthHeightDepth;
    while ((pos = string.find(delimiter)) != std::string::npos) {
        token = string.substr(0, pos);
        widthHeightDepth.push_back(std::stoi(token));
        string.erase(0, pos + delimiter.length());
    }

    this->selectDisparities(widthHeightDepth[2]);
    this->selectResolution(widthHeightDepth[0], widthHeightDepth[1]);
    this->selectFramerate(30);

    // Start stream
    bool status = cameraInterface->startStreams(crl::multisense::Source_Disparity);
    printf("Started stream| status: %d\n", status);

    this->modeChange = true;
}

void CRLPhysicalCamera::stop() {

}

void CRLPhysicalCamera::update(Base::Render render, crl::multisense::image::Header *pHeader) {


}

CRLBaseCamera::PointCloudData *CRLPhysicalCamera::getStream() {
    return meshData;
}

crl::multisense::image::Header CRLPhysicalCamera::getImage() {
    return imageP;
}

CRLPhysicalCamera::~CRLPhysicalCamera() {

    if (meshData->vertices != nullptr)
        free(meshData->vertices);

}

CRLBaseCamera::CameraInfo CRLPhysicalCamera::getInfo() {
    return CRLBaseCamera::cameraInfo;
}

// Pick an image size
crl::multisense::image::Config  CRLPhysicalCamera::getImageConfig() const {
    // Configure the sensor.
    crl::multisense::image::Config cfg;
    bool status = cameraInterface->getImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to query image config: %d\n", status);
    }

    return cfg;
}