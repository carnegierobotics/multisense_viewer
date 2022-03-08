//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLPhysicalCamera : CRLBaseCamera {
public:

    explicit CRLPhysicalCamera(CRLCameraDataType type) : CRLBaseCamera(type) {
        CRLBaseCamera::prepare();
    }

    std::string description;
    std::string data;

    int point = 0;

    void connect();
    void start() override;
    void update(Base::Render render);
    void stop() override;
    PointCloudData *getStream() override;

    ~CRLPhysicalCamera();

    ImageData *getImageData();

};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
