//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#define MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H

#include <string>

#ifdef VKRENDER_GIGEVISION_ENABLED

#include "Viewer/Modules/MultiSense/MultiSenseInterface.h"
#include "Viewer/Tools/Logger.h"
#include <libcrlgev/camera_obj.hh>


namespace VkRender::MultiSense {
    class GigEVisionConnector : public MultiSenseInterface {
    public:
        GigEVisionConnector() {
            Log::Logger::getInstance()->info("GigEVisionConnector Construct");
            m_lastEnumerateTime = std::chrono::steady_clock::now();
            m_gigEv = std::make_unique<device_obj>();

        }

        ~GigEVisionConnector() override {
            Log::Logger::getInstance()->info("GigEVisionConnector Destruct");

        }

        void getCameraInfo(MultiSenseProfileInfo *profileInfo) override {

        }

        void connect(std::string ip) override;
        void update(MultiSenseUpdateData* updateData) override;
        void setup() override;
        void disconnect() override;
        void getImage(MultiSenseStreamData* data) override;

        MultiSenseConnectionState connectionState() override {
            return MultiSenseInterface::connectionState();
        }

        struct Image{
            uint8_t* img;
            uint32_t imageSize;
            uint32_t width, height;


        };
    private:
        // Periodically enumerate devices
        std::chrono::time_point<std::chrono::steady_clock> m_lastEnumerateTime;


        std::unique_ptr<device_obj> m_gigEv;


        static void streamCallback(image_data info, uint8_t *img_buffer);
    };
}

#else
namespace VkRender::MultiSense {

    class GigEVisionConnector : public MultiSenseInterface {

    public:
        ~GigEVisionConnector() override = default;

        void connect(std::string ip) override {

        }

        void disconnect() override {

        }

        void update() override {

        }

        void setup() override {

        }

        uint8_t *getImage() override {
            return nullptr;
        }

        MultiSenseConnectionState connectionState() override {
            return MultiSenseInterface::connectionState();
        }
    };
}

#endif // GIGEV_ENABLED
#endif //MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
