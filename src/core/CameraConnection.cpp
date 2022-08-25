//
// Created by magnus on 3/21/22.
//



#include "CameraConnection.h"
#include <MultiSense/src/crl_camera/CRLVirtualCamera.h>
#include <MultiSense/src/tools/Logger.h>
CameraConnection::CameraConnection() {

}

void CameraConnection::updateActiveDevice(AR::Element dev) {
}

void CameraConnection::onUIUpdate(std::vector<AR::Element> *devices) {
    // If no device is connected then return
    if (devices == nullptr)
        return;

    // Check for actions on each element
    for (auto &dev: *devices) {
        // Connect if we click a device or if it is just added
        if ((dev.clicked && dev.state != AR_STATE_ACTIVE) || dev.state == AR_STATE_JUST_ADDED) {
            connectCrlCamera(dev);
            continue;
        }

        updateDeviceState(&dev);

        // Make sure inactive devices' preview are not drawn.
        if (dev.state != AR_STATE_ACTIVE) {
            for (auto &s: dev.streams)
                s.second.playbackStatus = AR_PREVIEW_NONE;
            continue;
        }


        updateActiveDevice(dev);

        // Disable if we click a device already connected
        if (dev.clicked && dev.state == AR_STATE_ACTIVE) {
            // Disable all streams
            for (auto &s: dev.streams)
                s.second.playbackStatus = AR_PREVIEW_RESET;

            disableCrlCamera(dev);
            continue;
        }
    }


}

void CameraConnection::connectCrlCamera(AR::Element &dev) {
    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    bool connected = false;

    Log::Logger::getInstance()->info("CameraConnection:: Connect.");


    if (dev.cameraName == "Virtual Camera") {
        camPtr = new CRLVirtualCamera();
        connected = camPtr->connect("None");
        if (connected) {
            dev.state = AR_STATE_ACTIVE;
            dev.cameraName = "Virtual Camera";
            dev.IP = "Local";
            lastActiveDevice = dev.name;

            AR::StreamingModes virtualCam{};
            virtualCam.name = "1. Virtual Left";
            virtualCam.sources.emplace_back("rowbot_short.mpg");
            virtualCam.sources.emplace_back("jeep_multisense_front_aux_image_color.mp4");
            virtualCam.sources.emplace_back("crl_jeep_multisense_front_left_image_rect.mp4");

            virtualCam.streamIndex = AR_PREVIEW_VIRTUAL_LEFT;
            std::string modeName = "1920x1080";
            virtualCam.modes.emplace_back(modeName);
            virtualCam.selectedStreamingMode = virtualCam.modes.front();
            virtualCam.selectedStreamingSource = virtualCam.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_LEFT] = virtualCam;

            AR::StreamingModes virtualRight{};
            virtualRight.name = "2. Virtual Right";
            virtualRight.sources.emplace_back("rowbot_short.mpg");
            virtualRight.sources.emplace_back("jeep_multisense_front_aux_image_color.mp4");
            virtualRight.sources.emplace_back("crl_jeep_multisense_front_left_image_rect.mp4");
            virtualRight.streamIndex = AR_PREVIEW_VIRTUAL_RIGHT;
            modeName = "1920x1080";
            virtualRight.modes.emplace_back(modeName);
            virtualRight.selectedStreamingMode = virtualRight.modes.front();
            virtualRight.selectedStreamingSource = virtualRight.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_RIGHT] = virtualRight;
            AR::StreamingModes aux{};
            aux.name = "3. Virtual Auxiliary";
            aux.sources.emplace_back("rowbot_short.mpg");
            aux.sources.emplace_back("jeep_multisense_front_aux_image_color.mp4");
            aux.sources.emplace_back("crl_jeep_multisense_front_left_image_rect.mp4");
            aux.streamIndex = AR_PREVIEW_VIRTUAL_AUX;
            modeName = "1920x1080";
            aux.modes.emplace_back(modeName);
            aux.selectedStreamingMode = aux.modes.front();
            aux.selectedStreamingSource = aux.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_AUX] = aux;

            AR::StreamingModes virutalPC{};
            virutalPC.name = "4. Virtual Point cloud";
            virutalPC.sources.emplace_back("depth");
            virutalPC.streamIndex = AR_PREVIEW_VIRTUAL_POINT_CLOUD;
            modeName = "1920x1080";
            virutalPC.modes.emplace_back(modeName);
            virutalPC.selectedStreamingMode = virutalPC.modes.front();
            virutalPC.selectedStreamingSource = virutalPC.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_POINT_CLOUD] = virutalPC;


            Log::Logger::getInstance()->info("CameraConnection:: Creating new Virtual Camera.");


        } else
            dev.state = AR_STATE_UNAVAILABLE;

    } else {
        setNetworkAdapterParameters(dev);
        Log::Logger::getInstance()->info("CameraConnection:: Creating new physical camera.");

        camPtr = new CRLPhysicalCamera();
        connected = camPtr->connect(dev.IP);
        if (connected) {
            dev.state = AR_STATE_ACTIVE;
            dev.cameraName = camPtr->getCameraInfo().devInfo.name;
            setStreamingModes(dev);
            lastActiveDevice = dev.name;

        } else {
            delete camPtr;
            dev.state = AR_STATE_UNAVAILABLE;
            lastActiveDevice = "";

        }
    }
}

void CameraConnection::setStreamingModes(AR::Element &dev) {
    // Find sources for each imager and and set these correspondly in element
    // Start with left
    // TODO USE camPtr to fetch these values dynamically
    AR::StreamingModes left{};
    left.name = "1. Left Sensor";
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

    AR::StreamingModes right{};
    right.name = "2. Right Sensor";
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

    AR::StreamingModes disparity{};
    disparity.name = "3. Disparity image";
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

    AR::StreamingModes auxiliary{};
    auxiliary.name = "4. Auxiliary Sensor";
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

    AR::StreamingModes pointCloud{};
    pointCloud.name = "5. Point Cloud";
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


}

void CameraConnection::setNetworkAdapterParameters(AR::Element &dev) {


}

void CameraConnection::updateDeviceState(AR::Element *dev) {

    dev->state = AR_STATE_UNAVAILABLE;

    // IF our clicked device is the one we already clicked
    if (dev->name == lastActiveDevice) {
        dev->state = AR_STATE_ACTIVE;
    }


}

void CameraConnection::disableCrlCamera(AR::Element &dev) {
    dev.state = AR_STATE_DISCONNECTED;
    lastActiveDevice = "";

    Log::Logger::getInstance()->info("CameraConnection:: Disconnecting profile %s using camera %s", dev.name.c_str(),
                                     dev.cameraName.c_str());

    // Free camPtr memory and point it to null for a reset.
    delete camPtr;
    camPtr = nullptr;

}

CameraConnection::~CameraConnection() {
    // Make sure delete the camPtr for physical cameras so we run destructor on the physical camera class which
    // stops all streams on the camera
    if (camPtr != nullptr)
        delete camPtr;
}
