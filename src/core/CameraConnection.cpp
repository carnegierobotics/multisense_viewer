//
// Created by magnus on 3/21/22.
//

#include <MultiSense/src/crl_camera/CRLVirtualCamera.h>
#include <MultiSense/src/tools/Logger.h>

#include "CameraConnection.h"

CameraConnection::CameraConnection() {

}

void CameraConnection::updateActiveDevice(Element dev) {

    for (auto d: dev.streams) {


        // TODO REMOVE
        if (d.second.playbackStatus == AR_PREVIEW_PLAYING) {
            //camPtr->start(d.second.selectedStreamingMode, d.second.selectedStreamingSource);
        }
    }

    //camPtr->start(dev.selectedStreamingMode, "Color Rectified Aux");


    if (dev.button && !camPreviewBar.active) {
        camPreviewBar.active = true;
    } else if (dev.button && !camPreviewBar.active) {
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

        // Make sure inactive devices' preview are not drawn.
        if (dev.state != ArActiveState){
            for (auto &s: dev.streams)
                    s.second.playbackStatus = AR_PREVIEW_NONE;
            continue;
        }


        updateActiveDevice(dev);

        // Disable if we click a device already connected
        if (dev.clicked && dev.state == ArActiveState) {
            // Disable all streams
            for (auto &s: dev.streams)
                s.second.playbackStatus = AR_PREVIEW_RESET;

            disableCrlCamera(dev);
            continue;
        }
    }


}

void CameraConnection::connectCrlCamera(Element &dev) {
    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    bool connected = false;

    Log::Logger::getInstance()->info("CameraConnection:: Connect.");


    if (dev.cameraName == "Virtual Camera") {
        camPtr = new CRLVirtualCamera();
        connected = camPtr->connect("None");
        if (connected) {
            dev.state = ArActiveState;
            dev.cameraName = "Virtual Camera";
            dev.IP = "Local";
            lastActiveDevice = dev.name;

            StreamingModes virtualCam{};
            virtualCam.sources.emplace_back("None");
            virtualCam.sources.emplace_back("earth");
            virtualCam.sources.emplace_back("pixels");
            virtualCam.streamIndex = AR_PREVIEW_VIRTUAL;
            std::string modeName = "1920x1080";
            virtualCam.modes.emplace_back(modeName);
            virtualCam.selectedStreamingMode = virtualCam.modes.front();
            virtualCam.selectedStreamingSource = virtualCam.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL] = virtualCam;

            Log::Logger::getInstance()->info("CameraConnection:: Creating new Virtual Camera.");


        } else
            dev.state = ArUnavailableState;

    } else {
        setNetworkAdapterParameters(dev);
        Log::Logger::getInstance()->info("CameraConnection:: Creating new physical camera.");

        camPtr = new CRLPhysicalCamera();
        connected = camPtr->connect(dev.IP);
        if (connected) {
            dev.state = ArActiveState;
            dev.cameraName = camPtr->getCameraInfo().devInfo.name;
            setStreamingModes(dev);

        } else
            dev.state = ArUnavailableState;
    }
}

void CameraConnection::setStreamingModes(Element &dev) {
    // Find sources for each imager and and set these correspondly in element
    // Start with left
    // TODO USE camPtr to fetch these values dynamically
    StreamingModes left{};
    left.sources.emplace_back("None");
    left.sources.emplace_back("Raw Left");
    left.sources.emplace_back("Luma Left");
    left.sources.emplace_back("Luma Rectified Left");
    left.sources.emplace_back("Compressed");
    left.streamIndex = AR_PREVIEW_LEFT;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";
        left.modes.emplace_back(modeName);
    }
    left.selectedStreamingMode = left.modes.front();
    left.selectedStreamingSource = left.sources.front();

    StreamingModes right{};
    right.sources.emplace_back("None");
    right.sources.emplace_back("Raw Right");
    right.sources.emplace_back("Luma Right");
    right.sources.emplace_back("Luma Rectified Right");
    right.sources.emplace_back("Compressed");
    right.streamIndex = AR_PREVIEW_RIGHT;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        right.modes.emplace_back(modeName);

    }
    right.selectedStreamingMode = right.modes.front();
    right.selectedStreamingSource = right.sources.front();

    StreamingModes disparity{};
    disparity.sources.emplace_back("None");
    disparity.sources.emplace_back("Disparity Left");
    disparity.sources.emplace_back("Disparity Cost");
    disparity.sources.emplace_back("Disparity Cost");
    disparity.streamIndex = AR_PREVIEW_DISPARITY;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        disparity.modes.emplace_back(modeName);

    }
    disparity.selectedStreamingMode = disparity.modes.front();
    disparity.selectedStreamingSource = disparity.sources.front();

    StreamingModes auxiliary{};
    auxiliary.sources.emplace_back("None");
    auxiliary.sources.emplace_back("Color Rectified Aux");
    auxiliary.sources.emplace_back("Luma Rectified Aux");
    auxiliary.sources.emplace_back("Color + Luma Rectified Aux");
    auxiliary.streamIndex = AR_PREVIEW_AUXILIARY;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        auxiliary.modes.emplace_back(modeName);

    }
    auxiliary.selectedStreamingMode = auxiliary.modes.front();
    auxiliary.selectedStreamingSource = auxiliary.sources.front();

    StreamingModes pointCloud{};
    pointCloud.sources.emplace_back("None");
    pointCloud.sources.emplace_back("S19");
    pointCloud.streamIndex = AR_PREVIEW_POINT_CLOUD;

    for (int i = 0; i < camPtr->getCameraInfo().supportedDeviceModes.size(); ++i) {
        auto mode = camPtr->getCameraInfo().supportedDeviceModes[i];
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";

        pointCloud.modes.emplace_back(modeName);

    }
    pointCloud.selectedStreamingMode = pointCloud.modes.front();
    pointCloud.selectedStreamingSource = pointCloud.sources.front();

    dev.streams[AR_PREVIEW_LEFT] = left;
    dev.streams[AR_PREVIEW_RIGHT] = right;
    dev.streams[AR_PREVIEW_DISPARITY] = disparity;
    dev.streams[AR_PREVIEW_AUXILIARY] = auxiliary;
    dev.streams[AR_PREVIEW_POINT_CLOUD] = pointCloud;

    Log::Logger::getInstance()->info("CameraConnection:: setting available streaming modes");


    lastActiveDevice = dev.name;
}

void CameraConnection::setNetworkAdapterParameters(Element &dev) {




}

void CameraConnection::updateDeviceState(Element *dev) {

    dev->state = ArUnavailableState;

    // IF our clicked device is the one we already clicked
    if (dev->name == lastActiveDevice) {
        dev->state = ArActiveState;
    }


}

void CameraConnection::disableCrlCamera(Element &dev) {
    dev.state = ArDisconnectedState;
    lastActiveDevice = "";

    Log::Logger::getInstance()->info("CameraConnection:: Disconnecting profile %s using camera %s", dev.name.c_str(), dev.cameraName.c_str());

    // Free camPtr memory and point it to null for a reset.
    delete camPtr;
    camPtr = nullptr;

}

CameraConnection::~CameraConnection(){
    // Make sure delete the camPtr for physical cameras so we run destructor on the physical camera class which
    // stops all streams on the camera
    if (camPtr != nullptr)
        delete camPtr;
}
