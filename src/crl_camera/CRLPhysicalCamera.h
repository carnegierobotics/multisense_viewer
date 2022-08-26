//
// Created by magnus on 3/1/22.
//

#ifndef MULTISENSE_CRLPHYSICALCAMERA_H
#define MULTISENSE_CRLPHYSICALCAMERA_H

#include <MultiSense/src/crl_camera/CRLBaseInterface.h>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <bitset>
#include <iostream>

class CRLPhysicalCamera : public CRLBaseInterface {
public:

    CRLPhysicalCamera() : CRLBaseInterface() {

    }

    ~CRLPhysicalCamera() override {
        // TODO FREE RESOURCES MEMBER VARIABLES
         stop("All");
        printf("BaseInterface destructor\n");

    }

    void update();


    bool connect(const std::string& ip) override;
    void start(std::string string, std::string dataSourceStr) override;
    void stop( std::string dataSourceStr) override;
    void updateCameraInfo() override;
    void getCameraStream(std::string stringSrc, crl::multisense::image::Header *stream,
                         crl::multisense::image::Header **stream2 = nullptr) override;
    CameraInfo getCameraInfo() override;
    void preparePointCloud(uint32_t i, uint32_t i1) override;


private:
    struct Image
    {
        uint16_t height{0}, width{0};
        uint32_t size{0};
        int64_t frame_id{0};
        crl::multisense::DataSource source{};
        const void *data{nullptr};
    };

    struct BufferPair
    {
        std::mutex swap_lock;
        crl::multisense::image::Header active, inactive;
        void *activeCBBuf{nullptr}, *inactiveCBBuf{nullptr};  // weird multisense BufferStream object, only for freeing reserved data later
        Image user_handle;

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
    std::vector<crl::multisense::DataSource> enabledSources;
    crl::multisense::Channel * cameraInterface{};
    std::unordered_map<crl::multisense::DataSource,BufferPair> buffers_;

    std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> imagePointers;
    glm::mat4 kInverseMatrix{};


    std::string dataSourceToString(unsigned int d);
    unsigned int stringToDataSource(const std::string &d);
    static void setDelayedPropertyThreadFunc(void * context);
    void addCallbacks();
    static void imageCallback(const crl::multisense::image::Header &header, void *userDataP);

    void streamCallback(const crl::multisense::image::Header &image);

    void setResolution(uint32_t width, uint32_t height, uint32_t depth);

    void setExposure(uint32_t exp) override;

    void setExposureParams(ExposureParams p) override;
    void setWhiteBalance(WhiteBalanceParams param) override;
    void setPostFilterStrength(float filterStrength) override;
    void setGamma(float gamma) override;
    void setFps(float fps) override;
    void setGain(float gain) override;

};


#endif //MULTISENSE_CRLPHYSICALCAMERA_H
