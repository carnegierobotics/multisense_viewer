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
class CameraConnection {
public:
    CameraConnection();

    ~CameraConnection();

    struct CamPreviewBar {
        bool active = false;
        uint32_t numQuads = 3;

    } camPreviewBar;

    /** @brief Handle to the current camera object */
    CRLBaseInterface *camPtr = nullptr;
    bool preview = false;
    std::string lastActiveDevice = "-1";

    void onUIUpdate(std::vector<AR::Element> *pVector);

private:
    int sd = -1;

    AR::Element *currentActiveDevice = nullptr; // handle to current device

    void updateActiveDevice(AR::Element *dev);

    void connectCrlCamera(AR::Element &element);

    void updateDeviceState(AR::Element *element);

    void disableCrlCamera(AR::Element &dev);

    bool setNetworkAdapterParameters(AR::Element &dev);

    void setStreamingModes(AR::Element &dev);

    std::string dataSourceToString(unsigned int d);

    void initCameraModes(std::vector<std::string> *modes, std::vector<crl::multisense::system::DeviceMode> vector);

    void filterAvailableSources(std::vector<std::string> *sources, std::vector<uint32_t> array);

    std::vector<uint32_t> maskArrayLeft = {
            {crl::multisense::Source_Raw_Left,
             crl::multisense::Source_Luma_Left,
             crl::multisense::Source_Luma_Rectified_Left,
             crl::multisense::Source_Chroma_Left,
             crl::multisense::Source_Jpeg_Left,
             crl::multisense::Source_Rgb_Left,
             crl::multisense::Source_Compressed_Left,
             crl::multisense::Source_Compressed_Rectified_Left,}
    };

    std::vector<uint32_t> maskArrayAux = {
            {crl::multisense::Source_Chroma_Rectified_Aux,
             crl::multisense::Source_Raw_Aux,
             crl::multisense::Source_Luma_Aux,
             crl::multisense::Source_Luma_Rectified_Aux,
             crl::multisense::Source_Chroma_Aux,
             crl::multisense::Source_Compressed_Aux,
             crl::multisense::Source_Compressed_Rectified_Aux}
    };

    std::vector<uint32_t> maskArrayDisparity = {
            {crl::multisense::Source_Disparity_Left,
             crl::multisense::Source_Disparity_Right,
             crl::multisense::Source_Disparity_Cost,
             crl::multisense::Source_Disparity_Aux,}
    };

    std::vector<uint32_t> maskArrayRight = {{
        crl::multisense::Source_Raw_Right,
                                             crl::multisense::Source_Luma_Right,
                                             crl::multisense::Source_Luma_Rectified_Right,
                                             crl::multisense::Source_Chroma_Right,
                                             crl::multisense::Source_Compressed_Right,
                                             crl::multisense::Source_Compressed_Rectified_Right,}
    };
};


#endif //MULTISENSE_CAMERACONNECTION_H
