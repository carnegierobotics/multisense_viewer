//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_LAYER_H
#define MULTISENSE_LAYER_H

#define IMGUI_INCLUDE_IMGUI_USER_H
#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS

#include <imgui.h>
#include <array>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>

#include "Viewer/Core/Definitions.h"
#include "Viewer/Core/KeyInput.h"
namespace VkRender {

    /**
     * @brief A UI Layer drawn by \refitem GuiManager.
     * To add an additional UI layer see \refitem LayerExample.
     */
    class Layer {

    public:

        virtual ~Layer() = default;

        /** @brief
         * Pure virtual must be overridden.
         * Called ONCE after UI have been initialized
         */
        virtual void onAttach() = 0;

        /** @brief
         * Pure virtual must be overridden.
         * Called ONCE before UI objects are destroyed
         */
        virtual void onDetach() = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called per frame, but before each script (\refitem Example) is updated
         * @param handles a UI object handle to collect user input
         */
        virtual void onUIRender(GuiObjectHandles *handles) = 0;

        /**
         * @brief Pure virtual must be overridden.
         * Called after draw command have been recorded, but before this frame has ended.
         * Can be used to prepare for next frame for instance
         */
        virtual void onFinishedRender() = 0;
    };


    /** @brief An initialized object needed to create a \refitem Device */
    struct EntryConnectDevice {
        std::string profileName = "MultiSense";
        std::string IP = "10.66.171.21";
        std::string interfaceName;
        std::string description;
        uint32_t interfaceIndex{};

        std::string cameraName;
        bool isRemoteHead = false;
        EntryConnectDevice() = default;

        EntryConnectDevice(std::string ip, std::string iName, std::string camera, uint32_t idx, std::string desc) : IP(
                std::move(ip)),
                                                                                                                    interfaceName(
                                                                                                                            std::move(
                                                                                                                                    iName)),
                                                                                                                    description(
                                                                                                                            std::move(
                                                                                                                                    desc)),
                                                                                                                    interfaceIndex(
                                                                                                                            idx),
                                                                                                                    cameraName(
                                                                                                                            std::move(
                                                                                                                                    camera)) {
            profileName.reserve(64);
            IP.reserve(16);
        }

        void reset() {
            profileName = "";
            IP = "";
            interfaceName = "";
            interfaceIndex = 0;
        }

        /**
         * @brief Utility function to check if the requested profile in \ref m_Entry is not conflicting with any of the previously connected devices in the sidebar
         * @param devices list of current devices
         * @param entry new connection to be added
         * @return true of we can add this new profile to list. False if not
         */
        bool ready(const std::vector<VkRender::Device> &devices, const EntryConnectDevice &entry) const {
            bool profileNameEmpty = entry.profileName.empty();
            bool profileNameTaken = false;
            bool IPEmpty = entry.IP.empty();
            bool adapterNameEmpty = entry.interfaceName.empty();

            bool AdapterAndIPInTaken = false;
            // Loop through devices and check that it doesn't exist already.
            for (auto &d: devices) {
                if (d.IP == entry.IP && d.interfaceName == entry.interfaceName) {
                    AdapterAndIPInTaken = true;
                    //Log::Logger::getInstance()->info("Ip {} on adapter {} already in use", entry.IP, entry.interfaceName);
                }
                if (d.name == entry.profileName) {
                    profileNameTaken = true;
                    //Log::Logger::getInstance()->info("Profile m_Name '{}' already taken", entry.profileName);
                }

            }
            bool ready = true;
            if (profileNameEmpty || profileNameTaken || IPEmpty || adapterNameEmpty || AdapterAndIPInTaken)
                ready = false;
            return ready;
        }

        std::vector<std::string> getNotReadyReasons(const std::vector<VkRender::Device> &devices, const EntryConnectDevice &entry){
            std::vector<std::string> errors;
            bool profileNameEmpty = entry.profileName.empty();
            bool IPEmpty = entry.IP.empty();
            bool adapterNameEmpty = entry.interfaceName.empty();
            // Loop through devices and check that it doesn't exist already.
            for (auto &d: devices) {
                if (d.IP == entry.IP && d.interfaceName == entry.interfaceName) {
                    errors.emplace_back("The IP address on the selected adapter is in use");
                }
                if (d.name == entry.profileName) {
                    errors.emplace_back("Profile name already in use");
                }

            }
            if (profileNameEmpty)
                errors.emplace_back("Profile name cannot be left blank");

            if (IPEmpty)
                errors.emplace_back("IP Address cannot be left blank");
            if (adapterNameEmpty)
                errors.emplace_back("No selected network adapter");

            return errors;

        }
    };

}

#endif //MULTISENSE_LAYER_H
