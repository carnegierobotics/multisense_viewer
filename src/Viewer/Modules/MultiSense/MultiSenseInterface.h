//
// Created by mgjer on 12/09/2024.
//

#ifndef MULTISENSEVIEWERINTERFACE_H
#define MULTISENSEVIEWERINTERFACE_H

#include "CommonHeader.h"

namespace VkRender::MultiSense
{
    class MultiSenseInterface
    {
    public:
        virtual ~MultiSenseInterface() = default;
        virtual void connect(std::string ip, std::string adapterName) = 0;
        virtual void disconnect() = 0;

        virtual MultiSenseConnectionState connectionState() { return MULTISENSE_DISCONNECTED;}

    };
}
#endif //MULTISENSEVIEWERINTERFACE_H
