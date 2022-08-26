//
// Created by magnus on 3/21/22.
//



#include "CameraConnection.h"
#include <MultiSense/src/crl_camera/CRLVirtualCamera.h>
#include <MultiSense/src/tools/Logger.h>

CameraConnection::CameraConnection() {

}

void CameraConnection::updateActiveDevice(AR::Element *dev) {

    if (!dev->parameters.initialized){
        auto* p = &dev->parameters;

        auto conf = camPtr->getCameraInfo().imgConf;

        {
            p->ep.exposure = conf.exposure();
            p->ep.autoExposure = conf.autoExposure();
            p->ep.exposureSource = conf.exposureSource();
            p->ep.autoExposureThresh = conf.autoExposureThresh();
            p->ep.autoExposureDecay = conf.autoExposureDecay();
            p->ep.autoExposureMax = conf.autoExposureMax();
            p->ep.autoExposureTargetIntensity = conf.autoExposureTargetIntensity();

            p->ep.autoExposureRoiHeight = conf.autoExposureRoiHeight();
            p->ep.autoExposureRoiWidth = conf.autoExposureRoiWidth();
            p->ep.autoExposureRoiX = conf.autoExposureRoiX();
            p->ep.autoExposureRoiY = conf.autoExposureRoiY();
        }

        p->gain = conf.gain();
        p->fps = conf.fps();
        p->gamma = conf.gamma();


        p->wb.autoWhiteBalance = conf.autoWhiteBalance();
        p->wb.autoWhiteBalanceDecay = conf.autoWhiteBalanceDecay();
        p->wb.autoWhiteBalanceThresh = conf.autoWhiteBalanceThresh();

        p->wb.whiteBalanceBlue = conf.whiteBalanceBlue();
        p->wb.whiteBalanceRed = conf.whiteBalanceRed();

        p->stereoPostFilterStrength = conf.stereoPostFilterStrength();

        auto cal = camPtr->getCameraInfo().camCal;

        p->initialized = true;
    }

    if (dev->parameters.update){
        auto p = dev->parameters;

        camPtr->setExposureParams(p.ep);
        camPtr->setWhiteBalance(p.wb);
        camPtr->setPostFilterStrength(p.stereoPostFilterStrength);
        camPtr->setGamma(p.gamma);
        camPtr->setFps(p.fps);
        camPtr->setGain(p.gain);

    }

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


        updateActiveDevice(&dev);

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
            virtualCam.sources.emplace_back("crl_jeep_multisense_front_left_image_rect.mp4");
            virtualCam.sources.emplace_back("jeep_disparity.mp4");

            virtualCam.streamIndex = AR_PREVIEW_VIRTUAL_LEFT;
            std::string modeName = "1920x1080";
            virtualCam.modes.emplace_back(modeName);
            virtualCam.selectedStreamingMode = virtualCam.modes.front();
            virtualCam.selectedStreamingSource = virtualCam.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_LEFT] = virtualCam;

            AR::StreamingModes virtualRight{};
            virtualRight.name = "2. Virtual Right";
            virtualRight.sources.emplace_back("rowbot_short.mpg");
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

    auto supportedModes = camPtr->getCameraInfo().supportedDeviceModes;

    AR::StreamingModes left{};
    left.name = "1. Left Sensor";
    left.streamIndex = AR_PREVIEW_LEFT;
    initCameraModes(&left.modes, supportedModes);
    filterAvailableSources(&left.sources, maskArrayLeft);
    left.selectedStreamingMode = left.modes.front();
    left.selectedStreamingSource = left.sources.front();

    AR::StreamingModes right{};
    right.name = "2. Right Sensor";
    right.streamIndex = AR_PREVIEW_RIGHT;
    initCameraModes(&right.modes, supportedModes);
    filterAvailableSources(&right.sources, maskArrayRight);
    right.selectedStreamingMode = right.modes.front();
    right.selectedStreamingSource = right.sources.front();

    AR::StreamingModes disparity{};
    disparity.name = "3. Disparity Image";
    disparity.streamIndex = AR_PREVIEW_DISPARITY;
    initCameraModes(&disparity.modes, supportedModes);
    filterAvailableSources(&disparity.sources, maskArrayDisparity);
    disparity.selectedStreamingMode = disparity.modes.front();
    disparity.selectedStreamingSource = disparity.sources.front();

    AR::StreamingModes aux{};
    aux.name = "4. Aux Sensor";
    aux.streamIndex = AR_PREVIEW_AUXILIARY;
    initCameraModes(&aux.modes, supportedModes);
    filterAvailableSources(&aux.sources, maskArrayAux);
    aux.selectedStreamingMode = aux.modes.front();
    aux.selectedStreamingSource = aux.sources.front();

    AR::StreamingModes pointCloud{};
    pointCloud.name = "5. Point Cloud";
    pointCloud.streamIndex = AR_PREVIEW_POINT_CLOUD;
    initCameraModes(&pointCloud.modes, supportedModes);
    filterAvailableSources(&pointCloud.sources, maskArrayDisparity);
    pointCloud.selectedStreamingMode = pointCloud.modes.front();
    pointCloud.selectedStreamingSource = pointCloud.sources.front();

    dev.streams[AR_PREVIEW_LEFT] = left;
    dev.streams[AR_PREVIEW_RIGHT] = right;
    dev.streams[AR_PREVIEW_DISPARITY] = disparity;
    dev.streams[AR_PREVIEW_AUXILIARY] = aux;
    dev.streams[AR_PREVIEW_POINT_CLOUD] = pointCloud;

    Log::Logger::getInstance()->info("CameraConnection:: setting available streaming modes");


}

void CameraConnection::initCameraModes(std::vector<std::string> *modes,
                                       std::vector<crl::multisense::system::DeviceMode> deviceModes) {
    for (auto mode : deviceModes) {
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";
        modes->emplace_back(modeName);
    }
}
void CameraConnection::filterAvailableSources(std::vector<std::string> *sources, std::vector<uint32_t> maskVec) {
    uint32_t bits = camPtr->getCameraInfo().supportedSources;

    for (auto mask: maskVec) {
        bool enabled = (bits & mask);
        if (enabled) {
            sources->emplace_back(dataSourceToString(mask));
        }
        std::cout << dataSourceToString(mask) << " " << ((bits & mask) ? "on\n" : "off\n");
    }
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

std::string CameraConnection::dataSourceToString(crl::multisense::DataSource d) {
    switch (d) {
        case crl::multisense::Source_Raw_Left:
            return "Raw Left";
        case crl::multisense::Source_Raw_Right:
            return "Raw Right";
        case crl::multisense::Source_Luma_Left:
            return "Luma Left";
        case crl::multisense::Source_Luma_Right:
            return "Luma Right";
        case crl::multisense::Source_Luma_Rectified_Left:
            return "Luma Rectified Left";
        case crl::multisense::Source_Luma_Rectified_Right:
            return "Luma Rectified Right";
        case crl::multisense::Source_Chroma_Left:
            return "Color Left";
        case crl::multisense::Source_Chroma_Right:
            return "Source Color Right";
        case crl::multisense::Source_Compressed_Right:
            return "Source Compressed Right";
        case crl::multisense::Source_Compressed_Rectified_Right:
            return "Source Compressed Rectified Right | Jpeg Left";
        case crl::multisense::Source_Disparity_Left:
            return "Disparity Left";
        case crl::multisense::Source_Disparity_Cost:
            return "Disparity Cost";
        case crl::multisense::Source_Disparity_Right:
            return "Disparity Right";
        case crl::multisense::Source_Rgb_Left:
            return "Source Rgb Left | Source Compressed Rectified Aux";
        case crl::multisense::Source_Compressed_Left:
            return "Source Compressed Left";
        case crl::multisense::Source_Compressed_Rectified_Left:
            return "Source Compressed Rectified Left";
        case crl::multisense::Source_Lidar_Scan:
            return "Source Lidar Scan";
        case crl::multisense::Source_Raw_Aux:
            return "Raw Aux";
        case crl::multisense::Source_Luma_Aux:
            return "Luma Aux";
        case crl::multisense::Source_Luma_Rectified_Aux:
            return "Luma Rectified Aux";
        case crl::multisense::Source_Chroma_Aux:
            return "Color Aux";
        case crl::multisense::Source_Chroma_Rectified_Aux:
            return "Color Rectified Aux";
        case crl::multisense::Source_Disparity_Aux:
            return "Disparity Aux";
        case crl::multisense::Source_Compressed_Aux:
            return "Source Compressed Aux";
        default:
            return "Unknown";
    }
}