//
// Created by magnus on 3/21/22.
//

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include <MultiSense/src/imgui/Layer.h>
#include <memory>
#include "MultiSense/external/simpleini/SimpleIni.h"
#include "ThreadPool.h"
#include "CRLPhysicalCamera.h"

#define NUM_THREADS 3

/**
 * Class handles the bridge between the GUI interaction and actual communication to camera
 * Also handles all configuration with local network adapter
 */
class CameraConnection {
public:
    CameraConnection(){
        pool = std::make_unique<ThreadPool>(1);
    }
    ~CameraConnection();

    std::unique_ptr<CRLPhysicalCamera> camPtr;
    std::unique_ptr<ThreadPool> pool;

    /**@brief Called once per frame with a handle to the devices UI information block **/
    void onUIUpdate(std::vector<MultiSense::Device> *pVector, bool shouldConfigNetwork, bool isRemoteHead);
    /**@brief Writes the current state of *dev to crl.ini configuration file **/
    void saveProfileAndDisconnect(MultiSense::Device *dev);
private:
    int sd = -1;
    std::mutex writeParametersMtx{};


    void updateActiveDevice(MultiSense::Device *dev);
    bool setNetworkAdapterParameters(MultiSense::Device &dev, bool b);
    void getStreamingModes(MultiSense::Device &dev);
    void initCameraModes(std::vector<std::string> *modes, std::vector<crl::multisense::system::DeviceMode> vector);
    void updateFromCameraParameters(MultiSense::Device *dev, uint32_t index) const;
    void filterAvailableSources(std::vector<std::string> *sources, std::vector<uint32_t> maskVec, uint32_t idx);

    // Add ini entry with log lines
    static void addIniEntry(CSimpleIniA* ini, std::string section, std::string key, std::string value);
    static void deleteIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value);

    // Caller functions for every *Task function is meant to be a threaded function
    static void setExposureTask(void * context, void* arg1, MultiSense::Device* dev);
    static void setWhiteBalanceTask(void * context, void* arg1, MultiSense::Device* dev);
    static void setLightingTask(void * context, void* arg1, MultiSense::Device* dev);
    static void setResolutionTask(void * context, CRLCameraResolution arg1, uint32_t idx);
    static void setAdditionalParametersTask(void *context, float fps, float gain, float gamma, float spfs,
                                            bool hdr, uint32_t index,
                                            MultiSense::Device *dev);
    static void connectCRLCameraTask(void* context, MultiSense::Device* dev, bool remoteHead, bool config);
    static void startStreamTask(void *context, MultiSense::Device *dev, std::string src, uint32_t remoteHeadIndex);
    static void stopStreamTask(void *context, MultiSense::Device *dev, std::string src, uint32_t remoteHeadIndex);

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
            crl::multisense::Source_Luma_Aux,
            crl::multisense::Source_Luma_Rectified_Aux,
            crl::multisense::Source_Chroma_Aux,
            crl::multisense::Source_Compressed_Aux,
            crl::multisense::Source_Compressed_Rectified_Aux
    };
};


#endif //MULTISENSE_CAMERACONNECTION_H
