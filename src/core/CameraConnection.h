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

    struct CamPreviewBar {
        bool active = false;
        uint32_t numQuads = 3;

    }camPreviewBar;

    /** @brief Handle to the current camera object */
    CRLBaseCamera *camPtr = nullptr;
    bool preview = false;
    //std::shared_ptr<CRLBaseCamera> camPtr;

    CameraConnection();
    void onUIUpdate(std::vector<Element> *pVector);

private:

    //CRLPhysicalCamera *prevCam; // Quick and dirty way of remebering old devices

    void updateActiveDevice(Element element);

    void connectCrlCamera(Element &element);
};


#endif //MULTISENSE_CAMERACONNECTION_H
