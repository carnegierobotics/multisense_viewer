//
// Created by magnus on 6/26/24.
//

#include "MultiSenseRendererBridge.h"

#include <utility>

#include "Viewer/Tools/AdapterUtils.h"
#include "Viewer/Modules/MultiSense/MultiSenseDevice.h"

namespace VkRender::MultiSense {

    void MultiSenseRendererBridge::update() {

        if (m_selectedProfileRef) {
            m_selectedProfileRef->multiSenseTaskManager->update();

            // enable streams and such
            auto& dev = m_selectedProfileRef->profileCreateInfo.deviceData();
            const auto& task = m_selectedProfileRef->multiSenseTaskManager;



            if (dev.hasSourceUpdate()) {
                // Collect all requested sources from the stream windows
                std::vector<std::string> requestedSources;
                for (auto& window : dev.streamWindow) {
                    requestedSources.emplace_back(window.second.enabledSource);
                    window.second.sourceUpdate = false; // Reset source update flag
                }
                // Get currently enabled sources
                std::vector<std::string>& enabledSources = task->getEnabledSources();
                // Determine which sources need to be stopped
                std::vector<std::string> sourcesToStop;
                for (const auto& enabledSource : enabledSources) {
                    if (std::find(requestedSources.begin(), requestedSources.end(), enabledSource) == requestedSources.end()) {
                        // If the enabled source is not in the requested sources, stop it
                        sourcesToStop.emplace_back(enabledSource);
                    }
                }
                // Determine which sources need to be started
                std::vector<std::string> sourcesToStart;
                for (const auto& requestedSource : requestedSources) {
                    if (std::find(enabledSources.begin(), enabledSources.end(), requestedSource) == enabledSources.end()) {
                        // If the requested source is not currently enabled, start it
                        sourcesToStart.emplace_back(requestedSource);
                    }
                }
                // Start the requested sources that aren't already enabled
                if (!sourcesToStart.empty()) {
                    task->startStreaming(sourcesToStart);
                }
                // Stop the sources that are enabled but not requested
                if (!sourcesToStop.empty()) {
                    task->stopStreaming(sourcesToStop);
                }
            }


        }

    }

    void MultiSenseRendererBridge::setSelectedMultiSenseProfile(std::shared_ptr<MultiSenseDevice> device) {
        m_selectedProfileRef = std::move(device);

    }

    void MultiSenseRendererBridge::setup() {
        // Initiate gigevision protocol and start searching
        // Create a profile in case we find one
        MultiSenseProfileInfo profileInfo;
        MultiSenseDevice device;
        profileInfo.profileName = "GigE-V Profile";
        profileInfo.connectionType = MULTISENSE_CONNECTION_TYPE_GIGEVISION;
        addNewProfile(profileInfo, false);

    }


    void MultiSenseRendererBridge::addNewProfile(MultiSenseProfileInfo createInfo, bool retrieveInfo) {
        if (m_multiSenseDevices.size() > 3)
            return;
        std::shared_ptr<MultiSenseDevice> device = std::make_shared<MultiSenseDevice>();
        switch (createInfo.connectionType) {
            case MULTISENSE_CONNECTION_TYPE_LIBMULTISENSE:
                device->multiSenseTaskManager = std::make_shared<MultiSenseTaskManager>(
                        MULTISENSE_CONNECTION_TYPE_LIBMULTISENSE);
                break;
            case MULTISENSE_CONNECTION_TYPE_GIGEVISION:
                device->multiSenseTaskManager = std::make_shared<MultiSenseTaskManager>(
                        MULTISENSE_CONNECTION_TYPE_GIGEVISION);
                break;
        }
        device->profileCreateInfo = std::move(createInfo);

        setSelectedMultiSenseProfile(device);
        m_multiSenseDevices.emplace_back(std::move(device));

        if (m_selectedProfileRef) {
            m_selectedProfileRef->connect();
            if (retrieveInfo)
                m_selectedProfileRef->retrieveCameraInfo();
        }
    }

    std::vector<std::shared_ptr<MultiSenseDevice>> MultiSenseRendererBridge::getAllMultiSenseProfiles() {

        return m_multiSenseDevices;
    }

    void MultiSenseRendererBridge::removeProfile(std::shared_ptr<MultiSenseDevice> profileToRemove) {
        // if its selected
        if (m_selectedProfileRef == profileToRemove) {
            m_selectedProfileRef = nullptr;
        }
        // remove it
        // erase-remove idiom

        m_multiSenseDevices.erase(
                std::remove_if(m_multiSenseDevices.begin(), m_multiSenseDevices.end(),
                               [&profileToRemove](const std::shared_ptr<MultiSenseDevice> &device) {
                                   return device->profileCreateInfo.profileName ==
                                          profileToRemove->profileCreateInfo.profileName;
                               }),
                m_multiSenseDevices.end()
        );

    }

    void MultiSenseRendererBridge::connect(bool retrieveCameraInfo) {
        if (m_selectedProfileRef) {
            m_selectedProfileRef->connect();
            if (retrieveCameraInfo){
                m_selectedProfileRef->retrieveCameraInfo();
            }
        }
    }

    void MultiSenseRendererBridge::disconnect() {
        if (m_selectedProfileRef) {
            m_selectedProfileRef->multiSenseTaskManager->disconnect();
        }
    }

    std::vector<std::string> MultiSenseRendererBridge::availableSources() {
        std::vector<std::string> sources;
        sources.emplace_back("No Source");
        sources.emplace_back("Luma Rectified Left");
        sources.emplace_back("Luma Rectified Right");
        sources.emplace_back("Disparity Left");
        return sources;
    }

    MultiSenseProfileInfo &MultiSenseRendererBridge::getSelectedMultiSenseProfile() {
        if (m_selectedProfileRef) {
            return m_selectedProfileRef->getCameraInfo();
        } else {
            throw std::runtime_error(
                    "No MultiSenseProfile Selected: 'getSelectedMultiSenseProfile'"); // TODO return default profile?
        }
    }

    bool MultiSenseRendererBridge::anyMultiSenseDeviceOnline() {
        bool anyConnected = false; // TODO set false
        for (const auto &dev: m_multiSenseDevices) {
            if (dev->multiSenseTaskManager->connectionState() == MultiSenseConnectionState::MULTISENSE_CONNECTED)
                anyConnected = true;
        }
        return anyConnected;
    }

    void MultiSenseRendererBridge::getImage(MultiSenseStreamData* data) {

        if (m_selectedProfileRef){
            m_selectedProfileRef->multiSenseTaskManager->getImage(data);
        }

    }
}
