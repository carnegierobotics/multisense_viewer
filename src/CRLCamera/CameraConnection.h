//
// Created by magnus on 3/21/22.
//

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include "CRLBaseInterface.h"
#include <MultiSense/src/imgui/Layer.h>
#include <memory>
#include "MultiSense/external/simpleini/SimpleIni.h"
#include "ThreadPool.h"
#define NUM_THREADS 3

/**
 * Class handles the bridge between the GUI interaction and actual communication to camera
 * Also handles all configuration with local network adapter
 */
class CameraConnection {
public:
    CameraConnection(){

        pool = std::make_unique<ThreadPool>(NUM_THREADS);

    }

    ~CameraConnection();

    /** @brief Handle to the current camera object */
    bool preview = false;
    std::string lastActiveDevice = "-1";
    void onUIUpdate(std::vector<AR::Element> *pVector);

    std::unique_ptr<CRLBaseInterface> camPtr;
    std::unique_ptr<ThreadPool> pool;

private:
    int sd = -1;
    std::mutex writeParametersMtx;

#ifdef WIN32
    unsigned long dwRetVal = 0;
    unsigned long NTEContext = 0;
#endif
    std::string hostAddress;

    void updateActiveDevice(AR::Element *dev);

    void connectCrlCamera(AR::Element &element);

    void updateDeviceState(AR::Element *element);

    void disableCrlCamera(AR::Element &dev);

    bool setNetworkAdapterParameters(AR::Element &dev);

    void setStreamingModes(AR::Element &dev);

    void initCameraModes(std::vector<std::string> *modes, std::vector<crl::multisense::system::DeviceMode> vector);

    void filterAvailableSources(std::vector<std::string> *sources, std::vector<uint32_t> array);

    static void addIniEntry(CSimpleIniA* ini, std::string section, std::string key, std::string value);

    std::vector<uint32_t> maskArrayAll = {
            crl::multisense::Source_Luma_Left,
            crl::multisense::Source_Luma_Rectified_Left,
            crl::multisense::Source_Compressed_Left,
            crl::multisense::Source_Compressed_Rectified_Left,
            crl::multisense::Source_Disparity_Left,

            crl::multisense::Source_Luma_Right,
            crl::multisense::Source_Luma_Rectified_Right,
            crl::multisense::Source_Compressed_Right,
            crl::multisense::Source_Compressed_Rectified_Right,

            crl::multisense::Source_Chroma_Rectified_Aux,
            crl::multisense::Source_Raw_Aux,
            crl::multisense::Source_Luma_Aux,
            crl::multisense::Source_Luma_Rectified_Aux,
            crl::multisense::Source_Chroma_Aux,
            crl::multisense::Source_Compressed_Aux,
            crl::multisense::Source_Compressed_Rectified_Aux
    };


    // Caller functions for threadpool
    static void setExposureTask(void * context, void* arg1, AR::Element* dev);
    static void setWhiteBalanceTask(void * context, void* arg1, AR::Element* dev);
    static void setLightingTask(void * context, void* arg1, AR::Element* dev);
    static void setAdditionalParametersTask(void * context, float fps, float gain, float gamma, float spfs, AR::Element* dev);
    static void connectCRLCameraTask(void* context, AR::Element* dev);
    static void disconnectCRLCameraTask(void* context, AR::Element* dev);

    void updateFromCameraParameters(AR::Element *dev) const;
};


#endif //MULTISENSE_CAMERACONNECTION_H
