//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#define MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#include <cstdint>
#include <string>
#include <vector>

#include <libcrlgev/camera_obj.hh>
#

namespace VkRender::MultiSense {
    class GigEVisionConnector {
    public:
        GigEVisionConnector(){


        }

        void initiate(){
            camDevice = std::make_unique<device_obj>();
        }

        void searchForDevices(){
            camDevice->enumerate();

        }

        std::unique_ptr<device_obj> camDevice;

    };
}


#endif //MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
