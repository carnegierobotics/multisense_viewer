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
        virtual void connect(std::string ip) = 0;
        virtual void disconnect() = 0;
        virtual void update() = 0;
        virtual void setup() = 0;
        virtual void getCameraInfo(MultiSenseProfileInfo* profileInfo) = 0;
        virtual uint8_t* getImage() = 0;

        virtual MultiSenseConnectionState connectionState() { return MULTISENSE_DISCONNECTED;}

    };
}
#endif //MULTISENSEVIEWERINTERFACE_H
