//
// Created by magnus on 6/26/24.
//

#ifndef COMMONHEADER_H
#define COMMONHEADER_H
#include <cstdint>
#include <string>
#include <vector>

namespace VkRender {
    typedef enum MultiSenseConnectionState {
        MULTISENSE_DISCONNECTED =                  0x00,
        MULTISENSE_CONNECTED =                     0x01,
        MULTISENSE_CONNECTION_IN_PROGRESS =        0x02,
        MULTISENSE_JUST_ADDED =                    0x04,
    } MultiSenseConnectionState;

    struct MultiSenseProfileInfo {
        std::string profileName = "Default profile name";
        std::string ethernetDescription = "Default ehternet";
        std::string cameraModel = "MultiSense Model";
        std::string serialNumber = "Serial Number";
        std::string inputIP = "10.66.171.21";

        MultiSenseProfileInfo() {
            // Reserve memory in case user inputs are long
            profileName.reserve(128);
            ethernetDescription.reserve(128);
            cameraModel.reserve(128);
            serialNumber.reserve(128);
            inputIP.reserve(24);
        }
    };

    struct sensorData {
        uint32_t exposure = 20000;
        uint32_t fps = 30;
        float gain = 2.2f;;
    };

    struct MultiSenseDevice {
        MultiSenseConnectionState connectionState = MULTISENSE_DISCONNECTED;

        MultiSenseProfileInfo createInfo;
    };
}

#endif //COMMONHEADER_H
