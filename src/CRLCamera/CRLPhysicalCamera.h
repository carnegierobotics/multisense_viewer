//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <bitset>
#include <iostream>
#include <cstdint>
#include <utility>
#include <memory>
#include "MultiSense/src/Core/Definitions.h"
#include "include/MultiSense/MultiSenseChannel.hh"

class ImageBufferWrapper {
public:
    ImageBufferWrapper(crl::multisense::Channel *driver,
                       crl::multisense::image::Header data) :
            driver_(driver),
            callbackBuffer_(driver->reserveCallbackBuffer()),
            data_(std::move(data)) {
    }
    ~ImageBufferWrapper() {
        if (driver_) {
            driver_->releaseCallbackBuffer(callbackBuffer_);
        }
    }
    [[nodiscard]] const crl::multisense::image::Header& data() const noexcept {
        return data_;
    }
    ImageBufferWrapper operator=(const ImageBufferWrapper &) = delete;
private:
    crl::multisense::Channel *driver_ = nullptr;
    void *callbackBuffer_;
    const crl::multisense::image::Header data_;
};

class ImageDataBase {
public:
    void updateImageBuffer(const std::shared_ptr<ImageBufferWrapper>& buf) {
        // Lock
        std::lock_guard<std::mutex> lock(mut);
        // replace latest data into image pointers
        imagePointersMap[0][buf->data().source] = buf;
    }
    // Question: making it a return statement initiates a copy? Pass by reference and return image pointer?
    std::shared_ptr<ImageBufferWrapper> getImageBuffer(uint32_t idx, crl::multisense::DataSource src) {
        std::lock_guard<std::mutex> lock(mut);
        return imagePointersMap[idx][src];
    }
private:
    std::mutex mut;
    std::unordered_map<uint32_t, std::unordered_map<crl::multisense::DataSource, std::shared_ptr<ImageBufferWrapper>>> imagePointersMap{};
};

class ChannelWrapper
{
public:
    explicit ChannelWrapper(const std::string &ipAddress, crl::multisense::RemoteHeadChannel remoteHeadChannel = -1):
            channelPtr_(crl::multisense::Channel::Create(ipAddress, remoteHeadChannel)){
        imageBuffer = new ImageDataBase(); //TODO Rename
    }
    ~ChannelWrapper(){
        delete imageBuffer;
        if (channelPtr_) {
            crl::multisense::Channel::Destroy(channelPtr_);
        }
    }
    crl::multisense::Channel* ptr() noexcept{
        return channelPtr_;
    }
    ImageDataBase* imageBuffer{};
    ChannelWrapper(const ChannelWrapper&) = delete;
    ChannelWrapper operator=(const ChannelWrapper&) = delete;
private:
    crl::multisense::Channel* channelPtr_ = nullptr;
};


class CRLPhysicalCamera {
public:

    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime{}; // Timer to log every second
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTimeImu{}; // Timer to log every second
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> callbackTime{}; // Timer to see how long ago the callback was called

    CRLPhysicalCamera() = default;

    ~CRLPhysicalCamera() {
        // TODO FREE RESOURCES MEMBER VARIABLES
        for (auto &ch: channelMap) {
            if (ch.second->ptr() != nullptr)
                stop("All", ch.first);
        }
    }

    struct CameraInfo {
        crl::multisense::system::DeviceInfo devInfo{};
        crl::multisense::image::Config imgConf{};
        crl::multisense::lighting::Config lightConf{};
        crl::multisense::system::NetworkConfig netConfig{};
        crl::multisense::system::VersionInfo versionInfo{};
        crl::multisense::image::Calibration camCal{};
        std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes{};
        crl::multisense::DataSource supportedSources{0};
        std::vector<uint8_t *> rawImages{};
        int sensorMTU = 0;
        crl::multisense::image::Calibration calibration{};
        glm::mat4 kInverseMatrix{};
    };

    // Functions to interface to the camera
    std::vector<uint32_t> connect(const std::string &ip, bool isRemoteHead);
    bool start(const std::string &dataSourceStr, uint32_t remoteHeadID);
    bool stop(const std::string &dataSourceStr, uint32_t idx);
    bool getCameraStream(std::string stringSrc, VkRender::TextureData *tex, uint32_t idx);
    void preparePointCloud(uint32_t i, uint32_t i1);
    void setResolution(CRLCameraResolution resolution, uint32_t channelID);
    void setExposureParams(ExposureParams p, uint32_t channelID);
    void setWhiteBalance(WhiteBalanceParams param, uint32_t channelID);
    void setLighting(LightingParams light, uint32_t channelID);
    void setPostFilterStrength(float filterStrength, uint32_t channelID);
    void setGamma(float gamma, uint32_t channelID);
    void setFps(float fps, uint32_t index);
    void setGain(float gain, uint32_t channelID);
    void setHDR(bool hdr, uint32_t channelID);
    void setMtu(uint32_t mtu, uint32_t channelID);

    CameraInfo getCameraInfo(uint32_t idx);

private:
    std::unordered_map<crl::multisense::RemoteHeadChannel, std::unique_ptr<ChannelWrapper>> channelMap{};
    std::unordered_map<uint32_t, CRLCameraResolution> currentResolutionMap{};
    std::unordered_map<uint32_t, CameraInfo> infoMap;
    std::mutex setCameraDataMutex;
    /**@brief Boolean to ensure the streamcallbacks called from LibMultiSense threads dont access class data while this class is being destroyed. It does happens once in a while */
    void addCallbacks(uint32_t idx);
    static void remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP);
    void updateCameraInfo(uint32_t idx);
};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
