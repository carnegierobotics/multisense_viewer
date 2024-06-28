//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSEINTERFACE_H
#define MULTISENSEINTERFACE_H
#include <cstdint>
#include <string>
#include <vector>

#include "CommonHeader.h"

namespace VkRender {
    class MultiSenseInterface {
    public:
        std::vector<std::string> getAvailableAdapterList();

        void setSelectedAdapter(std::string adapterName);

        void addNewProfile(MultiSenseProfileInfo);

        std::vector<MultiSenseDevice> &getProfileList();

        void update();

    private:
        std::vector<MultiSenseDevice> m_multisenseDevices;
    };
}


#endif //MULTISENSEINTERFACE_H
