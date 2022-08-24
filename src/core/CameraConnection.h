//
// Created by magnus on 3/21/22.
//

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include <MultiSense/src/crl_camera/CRLBaseInterface.h>
#include <MultiSense/src/imgui/Layer.h>

/**
 * Class handles the bridge between the GUI interaction and actual communication to camera
 * Also handles all configuration with local network adapter
 */
class CameraConnection
{
public:
    CameraConnection();
    ~CameraConnection();

    struct CamPreviewBar {
        bool active = false;
        uint32_t numQuads = 3;

    }camPreviewBar;

    /** @brief Handle to the current camera object */
    CRLBaseInterface *camPtr = nullptr;
    bool preview = false;
    std::string lastActiveDevice{};

    void onUIUpdate(std::vector<AR::Element> *pVector);

private:
    int sd = 0;

    AR::Element* currentActiveDevice = nullptr; // handle to current device

    void updateActiveDevice(AR::Element element);

    void connectCrlCamera(AR::Element &element);

    void updateDeviceState(AR::Element *element);

    void disableCrlCamera(AR::Element &element);

    void setNetworkAdapterParameters(AR::Element &dev);

    void setStreamingModes(AR::Element &dev);
};


#endif //MULTISENSE_CAMERACONNECTION_H
