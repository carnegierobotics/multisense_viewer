//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_VIEWER_COMMONHEADER_H
#define MULTISENSE_VIEWER_COMMONHEADER_H

#include <ranges>

#include "Viewer/Application/pch.h"
#include "Viewer/Rendering/Core/UUID.h"

namespace VkRender::MultiSense {

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

    struct PerWindowData {
        std::string enabledSource{};
        int selectedSourceIndex = 0;
        bool sourceUpdate = false;

    };

    struct DeviceData {
        std::vector<std::string> resolutions{};
        std::vector<std::string> sources{"Select Data Source"};

        std::unordered_map<UUID, PerWindowData> streamWindow;

        int exposure = 20000;
        float fps = 30;

        bool hasSourceUpdate() const {
            return std::any_of(streamWindow.begin(), streamWindow.end(), [](const auto &entry) {
                return entry.second.sourceUpdate;
            });
        }

    };

    struct MultiSenseProfileInfo {
        std::string profileName = "Default profile";
        std::string ifName = "Default Ethernet";
        std::string cameraModel = "MultiSense Model";
        std::string serialNumber = "Serial Number";
        std::string inputIP = "10.66.171.21";

        MultiSenseConnectionType connectionType = MultiSenseConnectionType::MULTISENSE_CONNECTION_TYPE_LIBMULTISENSE;

        DeviceData &deviceData() {
            return deviceDataGUI;
        }

        MultiSenseProfileInfo() {
            // Reserve memory in case user inputs are long
            profileName.reserve(128);
            ifName.reserve(128);
            cameraModel.reserve(128);
            serialNumber.reserve(128);
            inputIP.reserve(24);
        }

    private:
        DeviceData deviceDataGUI;
        DeviceData deviceDataNetwork;
    };


    struct MultiSenseStreamData {
        std::string dataSource;
        uint32_t imageSize = 0;
        uint8_t *imagePtr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t id = 0;
    };
}

#endif //COMMONHEADER_H
