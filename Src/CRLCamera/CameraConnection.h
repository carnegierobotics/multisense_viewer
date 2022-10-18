//
// Created by magnus on 3/21/22.
//

#ifndef MULTISENSE_CAMERACONNECTION_H
#define MULTISENSE_CAMERACONNECTION_H

#include <MultiSense/Src/imgui/Layer.h>
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
    CameraConnection() = default;

    ~CameraConnection();

    /**Pointer to actual camera object*/
    std::unique_ptr<CRLPhysicalCamera> camPtr;
    /**Pointer to thread-pool commonly used for UI blocking operations*/
    std::unique_ptr<ThreadPool> pool;

    /**@brief Called once per frame with a handle to the devices UI information block
     * @param devices vector of devices 1:1 relationship with elements shown in sidebar
     * @param shouldConfigNetwork if user have ticked the "configure network" checkbox
     * @param isRemoteHead if the connected device is a remote head, also selected by user
     */
    void onUIUpdate(std::vector<MultiSense::Device> *devices, bool shouldConfigNetwork, bool isRemoteHead);

    /**@brief Writes the current state of *dev to crl.ini configuration file
     * @param dev which profile to save to crl.ini
     */
    void saveProfileAndDisconnect(MultiSense::Device *dev);

private:
    /**@brief file descriptor to configure network settings on Linux */
    int sd = -1;
    /**@brief mutex to prevent multiple threads to communicate with camera.
     * could be omitted if threadpool will always consist of one thread */
    std::mutex writeParametersMtx{};

    /**
     * @brief Function called once per update by \refitem onUIUpdate if we have an active device
     * @param dev which profile this device is connected to
     */
    void updateActiveDevice(MultiSense::Device *dev);

    /**@brief Update system network settings if requested by user or autocorrect is chosen
     *
     * @param dev which profile is selected
     * @param b should configure network
     * @return if the network adapter were successfully configured
     */
    bool setNetworkAdapterParameters(MultiSense::Device &dev, bool b);

    /**@brief Get profile from .ini file if the serial number is recognized.
     * @param dev Which profile to update
     */
    void getProfileFromIni(MultiSense::Device &dev);

    /**@brief Create a user readable list of the possible camera modes*/
    void initCameraModes(std::vector<std::string> *modes, std::vector<crl::multisense::system::DeviceMode> vector);

    // Add ini entry with log lines
    /**@brief Add a .ini entry and log it*/
    static void addIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value);

    /**@brief Delete a .ini entry and log it*/
    static void deleteIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value);

    // Caller functions for every *Task function is meant to be a threaded function
    /**@brief static function given to the threadpool to configure exposure of the sensor.
     * @param context pointer to the calling context
     * @param arg1 pointer to exposure params block
     * @param dev Which profile to update
     * @param index Which remote-head to select
     * */
    static void setExposureTask(void *context, ExposureParams *arg1, MultiSense::Device *dev,
                                crl::multisense::RemoteHeadChannel remoteHeadIndex);

    /**@brief static function given to the threadpool to configure the white balance of the sensor.
     * @param context pointer to the calling context
     * @param arg1 pointer to WhiteBalanceParams params block
     * @param dev Which profile to update
     * @param index Which remote-head to select
     * */
    static void setWhiteBalanceTask(void *context, WhiteBalanceParams *arg1, MultiSense::Device *dev,
                                    crl::multisense::RemoteHeadChannel remoteHeadIndex);

    /**@brief static function given to the threadpool to configure lighting of the sensor.
     * @param context pointer to the calling context
     * @param arg1 pointer to the Lighting params block
     * @param dev Which profile to update
     * @param index Which remote-head to select
     * */
    static void setLightingTask(void *context, LightingParams *arg1, MultiSense::Device *dev,
                                crl::multisense::RemoteHeadChannel remoteHeadIndex);

    /**@brief static function given to the threadpool to configure exposure of the sensor.
     * @param context pointer to the calling context
     * @param arg1 What resolution to choose
     * @param index Which remote-head to select
     * */
    static void
    setResolutionTask(void *context, CRLCameraResolution arg1, crl::multisense::RemoteHeadChannel remoteHeadIndex);

    /**@brief Set parameters to the sensor. Grouped together as in the UI
     * @param context pointer to the callers context
     * @param fps framerate to request
     * @param gain gain value
     * @param gamma gamma value
     * @param spfs stereo post filter strength
     * @param hdr enable hdr?
     * @param dev Which profile to update
     * @param index Which remotehead to select
     */
    static void setAdditionalParametersTask(void *context, float fps, float gain, float gamma, float spfs,
                                            bool hdr, MultiSense::Device *dev, crl::multisense::RemoteHeadChannel index
    );

    /**@brief Task to connect a CRL camera
     * @param context pointer to the callers context
     * @param dev What profile to connect to
     * @param remoteHead boolean to connect to remote head
     * @param config boolean to determine if application should set network settings
     */
    static void connectCRLCameraTask(void *context, MultiSense::Device *dev, bool remoteHead, bool config);

    /**@brief Request to start a stream
     * @param context pointer to the callers context
     * @param src What source to request start
     * @param remoteHeadIndex id of remote head to select
     */
    static void startStreamTask(void *context, std::string src,
                                crl::multisense::RemoteHeadChannel remoteHeadIndex);

    /**@brief Request to stop a stream
     * @param context pointer to the callers context
     * @param src What source to request stop
     * @param remoteHeadIndex id of remote head to select
     */
    static void stopStreamTask(void *context, std::string src,
                               crl::multisense::RemoteHeadChannel remoteHeadIndex);

    /**@brief Update the UI block using the active information block from the physical camera
     * @param dev profile to update UI from
     * @param remoteHeadIndex id of remote head
     */
    void updateFromCameraParameters(MultiSense::Device *dev, crl::multisense::RemoteHeadChannel remoteHeadIndex) const;

    /**@brief Filter the unsupported sources defined by \ref maskArrayAll*/
    void filterAvailableSources(std::vector<std::string> *sources, std::vector<uint32_t> maskVec, uint32_t idx);

    /**@brief MaskArray to sort out unsupported streaming modes. Unsupported for this application*/
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
