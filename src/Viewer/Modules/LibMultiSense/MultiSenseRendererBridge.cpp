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

            // Get connection state
            device.connectionState = m_multiSenseTaskManager.connectionState();


        }
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
}
