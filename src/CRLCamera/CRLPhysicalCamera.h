//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <MultiSense/src/CRLCamera/CRLBaseInterface.h>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <bitset>
#include <iostream>

class CRLPhysicalCamera : public CRLBaseInterface {
public:

    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> startTime{}; // Timer to log every second
    std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> callbackTime{}; // Timer to see how long ago the callback was called

    CRLPhysicalCamera() = default;

    ~CRLPhysicalCamera() override {
        // TODO FREE RESOURCES MEMBER VARIABLES
        stop("All");
        crl::multisense::Channel::Destroy(cameraInterface);
        stopForDestruction = true;
    }


    bool connect(const std::string& ip) override;
    bool start(CRLCameraResolution resolution, std::string dataSourceStr) override;
    bool stop(std::string dataSourceStr) override;
    void updateCameraInfo() override;
    bool getCameraStream(ArEngine::YUVTexture *tex) override;
    bool getCameraStream(std::string stringSrc, ArEngine::TextureData *tex) override;

    CameraInfo getCameraInfo() override;
    void preparePointCloud(uint32_t i, uint32_t i1) override;


private:
    struct Image
    {
        uint16_t height{0}, width{0};
        uint32_t size{0};
        int64_t frame_id{0};
        crl::multisense::DataSource source{};
        const void *data{};
    };

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

    CameraInfo info{};
    std::vector<crl::multisense::DataSource> enabledSources{};
    crl::multisense::Channel * cameraInterface{};
    std::unordered_map<crl::multisense::DataSource,BufferPair> buffers_{};

    std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> imagePointers{};
    glm::mat4 kInverseMatrix{};
    CRLCameraResolution currentResolution{};

    /**@brief Boolean to ensure the streamcallbacks called from LibMultiSense threads dont access class data while this class is being destroyed. It does happens once in a while */
    bool stopForDestruction = false;

    void addCallbacks();
    static void imageCallback(const crl::multisense::image::Header &header, void *userDataP);

    void streamCallback(const crl::multisense::image::Header &image);

    void setExposure(uint32_t exp) override;
    void setExposureParams(ExposureParams p) override;
    void setWhiteBalance(WhiteBalanceParams param) override;
    void setPostFilterStrength(float filterStrength) override;
    void setGamma(float gamma) override;
    void setFps(float fps) override;
    void setGain(float gain) override;
    void setResolution(CRLCameraResolution resolution) override;
};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
