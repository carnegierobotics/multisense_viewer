//
// Created by magnus on 3/21/22.
//

#include "CameraConnection.h"

void CameraConnection::setup() {


}


void CameraConnection::update() {

}

void CameraConnection::onUIUpdate(GuiObjectHandles *uiHandle) {

    if (uiHandle->devices == nullptr)
        return;

    for (auto &dev: *uiHandle->devices) {
        if (dev.clicked && dev.state != ArActiveState) {

            // Connect to camera
            printf("Connecting\n");

            CRLPhysicalCamera camera;
            camera.connect();
            if (camera.online)
                dev.state = ArActiveState;
            else
                dev.state = ArUnavailableState;

        }
    }


}


void CameraConnection::draw(VkCommandBuffer commandBuffer, uint32_t i) {

}
