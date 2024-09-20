//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_VIEWER_COMMONHEADER_H
#define MULTISENSE_VIEWER_COMMONHEADER_H

#include <cstdint>
#include <string>
#include <memory>

namespace VkRender::MultiSense {
    class MultiSenseTaskManager;

    typedef enum MultiSenseConnectionType {
        MULTISENSE_CONNECTION_TYPE_LIBMULTISENSE,
        MULTISENSE_CONNECTION_TYPE_GIGEVISION
    } MultiSenseConnectionType;

    typedef enum MultiSenseConnectionState {
        MULTISENSE_DISCONNECTED = 0x00,
        MULTISENSE_CONNECTED = 0x01,
        MULTISENSE_CHANNEL_BUSY = 0x08,
        MULTISENSE_UNAVAILABLE = 0x0F,
    } MultiSenseConnectionState;

    struct MultiSenseProfileInfo {
        bool show = false;
        std::string profileName = "Default profile";
        std::string ifName = "Default Ethernet";
        std::string cameraModel = "MultiSense Model";
        std::string serialNumber = "Serial Number";
        std::string inputIP = "10.66.171.21";

        MultiSenseProfileInfo() {
            // Reserve memory in case user inputs are long
            profileName.reserve(128);
            ifName.reserve(128);
            cameraModel.reserve(128);
            serialNumber.reserve(128);
            inputIP.reserve(24);
        }
    };

    struct sensorData {
        uint32_t exposure = 20000;
        uint32_t fps = 30;
        float gain = 2.2f;
    };

    struct MultiSenseDevice {
        std::shared_ptr<MultiSenseTaskManager> multiSenseTaskManager;
        MultiSenseProfileInfo profileCreateInfo;
    };
}

#endif //COMMONHEADER_H
