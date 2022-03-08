//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLPhysicalCamera : CRLBaseCamera {
public:

    explicit CRLPhysicalCamera(CRLCameraDataType type) : CRLBaseCamera() {
        CRLBaseCamera::prepare();
    }

    std::string description;
    std::string data;
    bool online = false;

    int point = 0;

    void connect();
    void start(std::string string) override;
    void update(Base::Render render, crl::multisense::image::Header *pHeader);
    void stop() override;
    CameraInfo getInfo();
    PointCloudData *getStream() override;

    ~CRLPhysicalCamera();

    ImageData *getImageData();

    crl::multisense::image::Header getImage();
};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
