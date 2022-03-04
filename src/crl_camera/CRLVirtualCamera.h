//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_CRLVIRTUALCAMERA_H
#define MULTISENSE_CRLVIRTUALCAMERA_H


#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLVirtualCamera : CRLBaseCamera {
public:
    std::string description;
    std::string data;
    PointCloudData* meshData;
    ImageData* imageData;
    int point = 0;

    void initialize(CRLCameraDataType source);
    void start() override;
    void update(Base::Render render);
    void stop() override;
    PointCloudData *getStream() override;
    ImageData* getImageData();
    ~CRLVirtualCamera();

private:

};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
