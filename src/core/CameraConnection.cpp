//
// Created by magnus on 3/21/22.
//

#include "CameraConnection.h"

CameraConnection::CameraConnection(){

}


void CameraConnection::onUIUpdate(std::vector<Element> *devices) {
    // If no device is connected then retunr
    if (devices == nullptr)
        return;

    for (auto &dev: *devices) {
        if (dev.clicked && dev.state != ArActiveState) {
            // Connect to camera
            printf("Connecting\n");
            dev.camPtr = new CRLPhysicalCamera();
            dev.camPtr->connect();
            if (dev.camPtr->online){

                for (int i = 0; i < dev.camPtr->getInfo().supportedDeviceModes.size(); ++i) {
                    auto mode = dev.camPtr->getInfo().supportedDeviceModes[i];
                    std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                                           std::to_string(mode.disparities) + "x";

                    StreamingModes streamingModes;
                    streamingModes.modeName = modeName;
                    dev.modes.emplace_back(streamingModes);

                }
                dev.state = ArActiveState;
            }

            else
                dev.state = ArUnavailableState;

        }
        if (dev.depthImage){
            dev.camPtr->start(dev.selectedStreamingMode, "Disparity Left");
        }

        if (dev.colorImage){
            dev.camPtr->start(dev.selectedStreamingMode, "Color Rectified Aux");
        }

        if (dev.state == ArActiveState){
            activeDevice = dev;

        }

    }
}


