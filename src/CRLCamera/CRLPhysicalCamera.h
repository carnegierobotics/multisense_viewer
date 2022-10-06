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
#include "MultiSense/src/Core/Definitions.h"
#include "include/MultiSense/MultiSenseChannel.hh"


class CRLPhysicalCamera {
public:

    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime{}; // Timer to log every second
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTimeImu{}; // Timer to log every second
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> callbackTime{}; // Timer to see how long ago the callback was called

    CRLPhysicalCamera() = default;

    ~CRLPhysicalCamera() {
        // TODO FREE RESOURCES MEMBER VARIABLES
        for (auto& ch: channelMap) {
            //std::scoped_lock<std::mutex>(mutexMap.at(ch.first));
            stop("All", ch.first);
            crl::multisense::Channel::Destroy(ch.second);
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
    }info{};

    std::map<uint32_t, CameraInfo> infoMap;
    bool start(const std::string& dataSourceStr, uint32_t remoteHeadID);
    bool stop(const std::string& dataSourceStr, uint32_t idx);

    bool getCameraStream(std::string stringSrc, VkRender::TextureData *tex, uint32_t idx);
    bool getImuRotation(VkRender::Rotation *rot);

    void preparePointCloud(uint32_t i, uint32_t i1);
    void setResolution(CRLCameraResolution resolution, uint32_t i);
    void setExposureParams(ExposureParams p);
    void setWhiteBalance(WhiteBalanceParams param);
    void setLighting(LightingParams light);
    void setPostFilterStrength(float filterStrength);
    void setGamma(float gamma);
    void setFps(float fps, uint32_t index);
    void setGain(float gain);
    void setHDR(bool hdr);

    std::vector<uint32_t> connect(const std::string &ip, bool isRemoteHead);
    CameraInfo getCameraInfo(uint32_t idx);

private:
    struct Image
    {
        uint16_t height{0}, width{0};
        uint32_t size{0};
        int64_t frame_id{0};
        crl::multisense::DataSource source{};
        const void *data{};
    };

    struct RotationData {
        float roll = 0;
        float pitch = 0;
        float yaw = 0;
    } rotationData;
    std::mutex swap_lock{};

    struct BufferPair
    {
        std::mutex swap_lock{};
        crl::multisense::image::Header active{}, inactive{};
        void *activeCBBuf{nullptr}, *inactiveCBBuf{nullptr};  // weird multisense BufferStream object, only for freeing reserved data later
        Image user_handle{};

        void refresh()   // swap to latest if possible
        {
            std::scoped_lock lock(swap_lock);

            auto handleFromHeader = [](const auto &h) {
                return Image{static_cast<uint16_t>(h.height), static_cast<uint16_t>(h.width), h.imageLength, h.frameId, h.source, h.imageDataP};
            };

            if ((activeCBBuf == nullptr && inactiveCBBuf != nullptr) || // special case: first init
                (active.frameId < inactive.frameId))
            {
                std::swap(active, inactive);
                std::swap(activeCBBuf, inactiveCBBuf);
                user_handle = handleFromHeader(active);
            }
        }
    };

    std::vector<crl::multisense::DataSource> enabledSources{};
    std::map<uint32_t, crl::multisense::Channel *> channelMap{};
    std::unordered_map<crl::multisense::DataSource,BufferPair> buffers_{};
    std::map<uint32_t, std::unordered_map<crl::multisense::DataSource,BufferPair>> buffersMap{};
    std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> imagePointers{};
    std::map<uint32_t ,std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header>> imagePointersMap{};
    glm::mat4 kInverseMatrix{};
    std::unordered_map<uint32_t ,CRLCameraResolution> currentResolutionMap{};

    /**@brief Boolean to ensure the streamcallbacks called from LibMultiSense threads dont access class data while this class is being destroyed. It does happens once in a while */
    void addCallbacks(uint32_t idx);
    void streamCallbackRemoteHead(const crl::multisense::image::Header &image, uint32_t idx);
    static void imuCallback(const crl::multisense::imu::Header &header, void *userDataP);
    static void remoteHeadOneCallback(const crl::multisense::image::Header &header, void *userDataP);
    static void remoteHeadTwoCallback(const crl::multisense::image::Header &header, void *userDataP);
    static void remoteHeadThreeCallback(const crl::multisense::image::Header &header, void *userDataP);
    static void remoteHeadFourCallback(const crl::multisense::image::Header &header, void *userDataP);

    void updateCameraInfo(uint32_t idx);

    void setMtu(uint32_t mtu, uint32_t channelID);
};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
