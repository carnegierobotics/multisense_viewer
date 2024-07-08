//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#define MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
#include <cstdint>
#include <string>
#include <vector>

#ifdef VKRENDER_GIGEVISION_ENABLED
#include <libcrlgev/camera_obj.hh>
#endif

namespace VkRender::MultiSense {
    class MultiSenseRendererGigEVisionBridge {
    public:
        MultiSenseRendererGigEVisionBridge(){

            //device_obj cam_device;
            //cam_device.enumerate();

        }
    };
}


#endif //MULTISENSE_RENDERER_GIG_E_VISION_BRIDGE_H
