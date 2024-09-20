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
            //device.connectionState = device.multiSenseTaskManager->connectionState();

            device.multiSenseTaskManager->update();


        }
    }

    void MultiSenseRendererBridge::setup() {
        // Initiate gigevision protocol and start searching
        // Create a profile in case we find one
        MultiSenseProfileInfo profileInfo;
        MultiSenseDevice device;
        device.profileCreateInfo = MultiSenseProfileInfo();
        device.multiSenseTaskManager = std::make_shared<MultiSenseTaskManager>(MULTISENSE_CONNECTION_TYPE_GIGEVISION);
        device.multiSenseTaskManager->setup();

        m_multiSenseDevices.emplace_back(device);
        //m_multiSenseTaskManager.setup();

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
        device.profileCreateInfo = std::move(createInfo);
        device.multiSenseTaskManager = std::make_shared<MultiSenseTaskManager>(MULTISENSE_CONNECTION_TYPE_LIBMULTISENSE);

        m_multiSenseDevices.emplace_back(device);
    }

    std::vector<MultiSenseDevice> &MultiSenseRendererBridge::getProfileList() {

        return m_multiSenseDevices;
    }

    void MultiSenseRendererBridge::removeProfile(const MultiSenseDevice &profile) {
        // erase-remove idiom
        m_multiSenseDevices.erase(
                std::remove_if(m_multiSenseDevices.begin(), m_multiSenseDevices.end(),
                               [&profile](const MultiSenseDevice &device) {
                                   return device.profileCreateInfo.profileName == profile.profileCreateInfo.profileName;
                               }),
                m_multiSenseDevices.end()
        );
    }

    void MultiSenseRendererBridge::connect(const MultiSenseDevice &profile) {
        m_multiSenseDevices.front().multiSenseTaskManager->connect(MultiSenseDevice());
    }

    void MultiSenseRendererBridge::disconnect(const MultiSenseDevice &profile) {
        // Check that we are not busy connection already of we are on the same device?
        m_multiSenseDevices.front().multiSenseTaskManager->disconnect();
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
            if (dev.multiSenseTaskManager->connectionState() == MultiSenseConnectionState::MULTISENSE_CONNECTED)
                anyConnected = true;
        }
        return anyConnected;
    }

    uint8_t *MultiSenseRendererBridge::getImage() {
        for (const auto &dev: m_multiSenseDevices) {
            if (dev.multiSenseTaskManager->connectionState() == MultiSenseConnectionState::MULTISENSE_CONNECTED){


            }
        return dev.multiSenseTaskManager->getImage();
        }
        return nullptr;
    }
}
