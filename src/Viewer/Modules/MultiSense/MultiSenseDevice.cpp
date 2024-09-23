//
// Created by magnus on 9/23/24.
//

#include "MultiSenseDevice.h"

#include "Viewer/Modules/MultiSense/MultiSenseTaskManager.h"

namespace VkRender::MultiSense{
    void MultiSenseDevice::connect() {
        std::lock_guard<std::mutex> lock(m_profileInfoMutex);
        multiSenseTaskManager->connect(profileCreateInfo);
    }

    void MultiSenseDevice::retrieveCameraInfo(){
        std::lock_guard<std::mutex> lock(m_profileInfoMutex);
        multiSenseTaskManager->retrieveCameraInfo(&profileCreateInfo);
    }
    // TODO maintain two copied. One which is updated and one which is always ready to read with imgui
    MultiSenseProfileInfo MultiSenseDevice::getCameraInfo(){
        std::lock_guard<std::mutex> lock(m_profileInfoMutex);
        return profileCreateInfo;
    }

}
