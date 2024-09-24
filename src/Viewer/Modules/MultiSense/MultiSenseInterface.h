//
// Created by mgjer on 12/09/2024.
//

#ifndef MULTISENSEVIEWERINTERFACE_H
#define MULTISENSEVIEWERINTERFACE_H

#include "CommonHeader.h"

namespace VkRender::MultiSense
{
    struct MultiSenseUpdateData {
        std::vector<std::string> enabledSources;
    };

    class MultiSenseInterface
    {
    public:
        virtual ~MultiSenseInterface() = default;
        virtual void connect(std::string ip) = 0;
        virtual void disconnect() = 0;
        virtual void update(MultiSenseUpdateData* updateData) = 0;
        virtual void setup() = 0;
        virtual void getCameraInfo(MultiSenseProfileInfo* profileInfo) = 0;
        virtual void getImage(MultiSenseStreamData* data) = 0;

        virtual void startStreaming(const std::vector<std::string>& streams) {};
        virtual void stopStream(const std::string& stream) {};

        virtual MultiSenseConnectionState connectionState() { return MULTISENSE_DISCONNECTED;}

    };
}
#endif //MULTISENSEVIEWERINTERFACE_H
