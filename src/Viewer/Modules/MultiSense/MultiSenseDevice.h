//
// Created by magnus on 9/23/24.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEDEVICE_H
#define MULTISENSE_VIEWER_MULTISENSEDEVICE_H

#include "Viewer/Modules/MultiSense/CommonHeader.h"

namespace VkRender::MultiSense {
    class MultiSenseTaskManager;

    class MultiSenseDevice {
    public:

        // Constructor to initialize the object
        MultiSenseDevice() = default;
        // Delete copy constructor and assignment operator to prevent copies (We cant copy a mutex)
        MultiSenseDevice(const MultiSenseDevice&) = delete;
        MultiSenseDevice& operator=(const MultiSenseDevice&) = delete;
        // Allow move constructor and move assignment operator
        MultiSenseDevice(MultiSenseDevice&&) = default;
        MultiSenseDevice& operator=(MultiSenseDevice&&) = default;

        std::shared_ptr<MultiSenseTaskManager> multiSenseTaskManager;
        MultiSenseProfileInfo profileCreateInfo;
        std::mutex m_profileInfoMutex;  // Mutex to guard profileInfo

        void connect();

        void retrieveCameraInfo();
        MultiSenseProfileInfo getCameraInfo();
    };


}
#endif //MULTISENSE_VIEWER_MULTISENSEDEVICE_H
