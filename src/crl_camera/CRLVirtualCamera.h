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
    MeshData* meshData;

    void initialize() override;
    void start() override;
    void update();
    void stop() override;
    MeshData *getStream() override;

    ~CRLVirtualCamera();

private:

};


#endif //MULTISENSE_CRLVIRTUALCAMERA_H
