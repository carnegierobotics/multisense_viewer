//
// Created by magnus on 3/21/22.
//

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>

class CameraConnection
{
public:

    CameraConnection();
    void onUIUpdate(std::vector<Element> *pVector);

    Element activeDevice;

};


#endif //MULTISENSE_CAMERACONNECTION_H
