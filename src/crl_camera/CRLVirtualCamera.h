//
// Created by magnus on 2/21/22.
//

#ifndef MULTISENSE_CRLVIRTUALCAMERA_H
#define MULTISENSE_CRLVIRTUALCAMERA_H


#include <MultiSense/src/crl_camera/CRLBaseCamera.h>

class CRLVirtualCamera : CRLBaseCamera {
public:
    explicit CRLVirtualCamera(CRLCameraDataType type) : CRLBaseCamera() {


    }
    std::string description;
    std::string data;
    int point = 0;

    void connect(CRLCameraDataType source);
    void start(std::string string, std::string dataSourceStr) override;
    void update(Base::Render render);
    void stop(std::string dataSourceStr) override;
    PointCloudData *getStream() override;

    ~CRLVirtualCamera();

private:

};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
