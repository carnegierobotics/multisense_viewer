//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSEINTERFACE_H
#define MULTISENSEINTERFACE_H
#include <cstdint>
#include <string>
#include <vector>

#include "Viewer/Tools/AdapterUtils.h"
#include "MultiSenseTaskManager.h"
#include "CommonHeader.h"

namespace VkRender::MultiSense {
    class MultiSenseRendererBridge {
    public:
        std::vector<std::string> getAvailableAdapterList();

        void setSelectedAdapter(std::string adapterName);

        void addNewProfile(MultiSenseProfileInfo);
        void removeProfile(const MultiSenseDevice& profile);
        void connect(const MultiSenseDevice& profile);
        void disconnect(const MultiSenseDevice& profile);

        std::vector<MultiSenseDevice> &getProfileList();

        std::vector<std::string> availableSources();

        void update();
        void setup();
        bool anyMultiSenseDeviceOnline();

    private:
        std::vector<MultiSenseDevice> m_multiSenseDevices;
        AdapterUtils m_adapterUtils;
        MultiSenseTaskManager m_multiSenseTaskManager;
    };
}


#endif //MULTISENSEINTERFACE_H
