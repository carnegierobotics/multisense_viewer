//
// Created by magnus on 6/26/24.
//

#include "MultiSenseRendererBridge.h"

#include <utility>

#include "Viewer/Tools/AdapterUtils.h"

namespace VkRender::MultiSense {

    void MultiSenseRendererBridge::update() {
        // check for new connection
        for (auto &device: m_multiSenseDevices) {
            // Get connection state from libmultisense
            device.connectionState = m_multiSenseTaskManager.connectionState();
        }
    }

    void MultiSenseRendererBridge::setup() {
        // Initiate gigevision protocol
        m_multiSenseTaskManager.initiateGigEV();
    }

    std::vector<std::string> MultiSenseRendererBridge::getAvailableAdapterList() {
        auto adapterList = m_adapterUtils.getAdapterList();
        std::vector<std::string> adapters;
        adapters.reserve(adapterList.size());
        for (const auto &adapter: adapterList) {
            adapters.emplace_back(adapter.ifName);
        }
        return adapters;
    }

    void MultiSenseRendererBridge::setSelectedAdapter(std::string adapterName) {

    }

    void MultiSenseRendererBridge::addNewProfile(MultiSenseProfileInfo createInfo) {
        if (m_multiSenseDevices.size() > 3)
            return;

        MultiSenseDevice device;
        // Perform connection with CRLPhysicalCamera and all that
        // Provide status updates to UI
        device.createInfo = std::move(createInfo);
        m_multiSenseDevices.emplace_back(device);
        m_multiSenseTaskManager.connect(device);
    }

    std::vector<MultiSenseDevice> &MultiSenseRendererBridge::getProfileList() {

        return m_multiSenseDevices;
    }

    void MultiSenseRendererBridge::removeProfile(const MultiSenseDevice &profile) {
        // erase-remove idiom
        m_multiSenseDevices.erase(
                std::remove_if(m_multiSenseDevices.begin(), m_multiSenseDevices.end(),
                               [&profile](const MultiSenseDevice &device) {
                                   return device.createInfo.profileName == profile.createInfo.profileName;
                               }),
                m_multiSenseDevices.end()
        );
    }

    void MultiSenseRendererBridge::connect(const MultiSenseDevice &profile) {
        // Check that we are not busy connection already of we are on the same device?
        m_multiSenseTaskManager.connect(profile);
    }

    void MultiSenseRendererBridge::disconnect(const MultiSenseDevice &profile) {
        // Check that we are not busy connection already of we are on the same device?
        m_multiSenseTaskManager.disconnect();
    }

    std::vector<std::string> MultiSenseRendererBridge::availableSources() {
        std::vector<std::string> sources;
        sources.emplace_back("No Source");
        sources.emplace_back("Luma Rectified Left");
        sources.emplace_back("Luma Rectified Right");
        sources.emplace_back("Disparity Left");
        return sources;
    }

    bool MultiSenseRendererBridge::anyMultiSenseDeviceOnline() {
        bool anyConnected = true; // TODO set false
        for (const auto &dev: m_multiSenseDevices) {
            if (dev.connectionState == MultiSenseConnectionState::MULTISENSE_CONNECTED)
                anyConnected = true;
        }
        return anyConnected;
    }
}
