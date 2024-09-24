//
// Created by magnus on 9/23/24.
//

#include "MultiSenseDevice.h"

#include "Viewer/Modules/MultiSense/MultiSenseTaskManager.h"

namespace VkRender::MultiSense{
    void MultiSenseDevice::connect() {
        multiSenseTaskManager->connect(profileCreateInfo);
    }

    void MultiSenseDevice::retrieveCameraInfo(){
        multiSenseTaskManager->retrieveCameraInfo(&profileCreateInfo);
    }
    // TODO maintain two copied. One which is updated and one which is always ready to read with imgui
    MultiSenseProfileInfo& MultiSenseDevice::getCameraInfo(){
        return profileCreateInfo;
    }

}
