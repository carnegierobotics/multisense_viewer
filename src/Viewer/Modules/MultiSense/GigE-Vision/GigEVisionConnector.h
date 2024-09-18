//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#define MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#include <string>

#include "Viewer/Modules/MultiSense/MultiSenseInterface.h"

#include <libcrlgev/camera_obj.hh>


namespace VkRender::MultiSense {
    class GigEVisionConnector : public MultiSenseInterface {
    public:
        GigEVisionConnector() = default;

        void connect(std::string ip, std::string ifName) override;

        void disconnect() override;

        void initiate(){
            camDevice = std::make_unique<device_obj>();
        }

        ~GigEVisionConnector() override = default;


        MultiSenseConnectionState connectionState() override {
            return MultiSenseInterface::connectionState();
        }

        void searchForDevices(){
            camDevice->enumerate();

        }

        std::unique_ptr<device_obj> camDevice;

    };
}


#endif //MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
