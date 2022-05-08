//
// Created by magnus on 3/21/22.
//

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include <MultiSense/src/crl_camera/CRLPhysicalCamera.h>
#include <MultiSense/src/imgui/Layer.h>

/**
 * Class handles the bridge between the GUI interaction and actual communication to camera
 */
class CameraConnection
{
public:

    CameraConnection();
    void onUIUpdate(std::vector<Element> *pVector);

    /** @brief Handle to the current camera object */
    CRLPhysicalCamera *camPtr = nullptr;
    bool preview = false;

    void updateActiveDevice(Element element);
};


#endif //MULTISENSE_CAMERACONNECTION_H
