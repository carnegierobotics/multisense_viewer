//
// Created by magnus on 6/26/24.
//

#include "MultiSenseInterface.h"

#include <utility>


namespace VkRender {

    void MultiSenseInterface::update() {
        // check for new connection
        for (auto& device: m_multisenseDevices) {
            if (device.connectionState == MULTISENSE_JUST_ADDED) {
                // Set this one to connecting state
                device.connectionState = MULTISENSE_CONNECTION_IN_PROGRESS;

                // Attempt connection
            }

            if (device.connectionState == MULTISENSE_CONNECTION_IN_PROGRESS) {

                // Check for results
                // If great then activate
                device.connectionState = MULTISENSE_CONNECTED;
            }
        }
    }

    std::vector<std::string> MultiSenseInterface::getAvailableAdapterList() {
            return {"ifname", "Ethernet2"};
    }

    void MultiSenseInterface::setSelectedAdapter(std::string adapterName) {

    }

    void MultiSenseInterface::addNewProfile(MultiSenseProfileInfo createInfo) {
        if (m_multisenseDevices.size() > 3)
            return;
        MultiSenseDevice device;
        // Perform connection with CRLPhysicalCamera and all that
        // Provide status updates to UI
        device.createInfo = std::move(createInfo);
        device.connectionState = MULTISENSE_JUST_ADDED;

        m_multisenseDevices.emplace_back(device);
    }

    std::vector<MultiSenseDevice>& MultiSenseInterface::getProfileList() {

        return m_multisenseDevices;
    }
}
