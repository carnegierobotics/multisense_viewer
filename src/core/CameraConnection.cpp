//
// Created by magnus on 3/21/22.
//

#include <MultiSense/src/crl_camera/CRLVirtualCamera.h>
#include "CameraConnection.h"

CameraConnection::CameraConnection() {

}

void CameraConnection::updateActiveDevice(Element dev) {

    if (dev.depthImage) {
        camPtr->start(dev.selectedStreamingMode, "Disparity Left");
    } else {
        camPtr->stop("Disparity Left");
    }


    if (dev.colorImage) {
        camPtr->start(dev.selectedStreamingMode, "Color Rectified Aux");
    } else {
        camPtr->stop("Color Rectified Aux");
    }

    if (dev.btnShowPreviewBar && !camPreviewBar.active) {

        camPreviewBar.active = true;
    } else if (dev.btnShowPreviewBar && !camPreviewBar.active) {
        camPreviewBar.active = false;
    }


}

void CameraConnection::onUIUpdate(std::vector<Element> *devices) {
    // If no device is connected then return
    if (devices == nullptr)
        return;

    // Check for actions on each element
    for (auto &dev: *devices) {
        // Connect if we click a device or if it is just added
        if ((dev.clicked && dev.state != ArActiveState) || dev.state == ArJustAddedState) {
            connectCrlCamera(dev);
            continue;
        }

        updateDeviceState(&dev);
        if (dev.state != ArActiveState)
            continue;


        updateActiveDevice(dev);

        // Disable if we click a device already connected
        if (dev.clicked && dev.state == ArActiveState){
            disableCrlCamera(dev);
            continue;
        }
    }


}

void CameraConnection::connectCrlCamera(Element &dev) {
    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    bool connected = false;

    if (dev.cameraName == "Virtual Camera") {
        camPtr = new CRLVirtualCamera();
        connected = camPtr->connect("None");
        if (connected) {
            dev.state = ArActiveState;
            dev.cameraName = "Virtual Camera";
            dev.IP = "Local";
        } else
            dev.state = ArUnavailableState;

    } else {
        camPtr = new CRLPhysicalCamera();
        connected = camPtr->connect(dev.IP);
        if (connected) {
            dev.state = ArActiveState;
            dev.cameraName = camPtr->getCameraInfo().devInfo.name;

            dev.modes.clear(); // Clear possible modes. Modes can maybe be dynamic between connections. If the camera's FW was updated in-between?

            for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
                auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
                std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                                       std::to_string(mode.disparities) + "x";

                StreamingModes streamingModes;
                streamingModes.modeName = modeName;
                dev.modes.emplace_back(streamingModes);
            }
        }
        else
            dev.state = ArUnavailableState;
    }

    currentActiveDevice = &dev;
    lastActiveDevice = dev.name;

}

void CameraConnection::updateDeviceState(Element* dev) {

    dev->state = ArUnavailableState;

    // IF our clicked device is the one we already clicked
    if (dev->name == lastActiveDevice){
        dev->state = ArActiveState;
    }


}

void CameraConnection::disableCrlCamera(Element &dev) {
    dev.state = ArDisconnectedState;
    lastActiveDevice = "";

    dev.colorImage = false;
    dev.colorImage = false;
    dev.depthImage = false;
    dev.pointCloud = false;

    // Free camPtr memory and point it to null for a reset.
    delete camPtr;
    camPtr = nullptr;

}


