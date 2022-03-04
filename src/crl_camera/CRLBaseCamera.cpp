//
// Created by magnus on 3/4/22.
//

#include "MultiSense/src/crl_camera/CRLBaseCamera.h"

void CRLBaseCamera::connect(std::string& hostname) {


    cameraInterface = crl::multisense::Channel::Create(hostname);


    if (cameraInterface != nullptr){
        cameraInterface->getDeviceInfo(devInfo);
        printf("DevInfo %s\n", devInfo.name.c_str());
    }

}

void CRLBaseCamera::prepare() {
    connect(DEFAULT_CAMERA_IP);

    imageData = new ImageData();
    meshData = new PointCloudData(1280, 720);

}
