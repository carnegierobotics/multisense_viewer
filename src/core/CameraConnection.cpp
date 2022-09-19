//
// Created by magnus on 3/21/22.
//

#include "CameraConnection.h"


#ifdef WIN32
#define _WINSOCKAPI_    // stops windows.h including winsock.h

#include <WinSock2.h>
#include <WS2tcpip.h>
#include <Windows.h>
#include <ipmib.h>
#include <iphlpapi.h> 

#define MALLOC(x) HeapAlloc(GetProcessHeap(), 0, (x))
#define FREE(x) HeapFree(GetProcessHeap(), 0, (x))
#define ADAPTER_HEX_NAME_LENGTH 38
#define UNNAMED_ADAPTER "Unnamed"
#else


#include <linux/if_ether.h>
#include <netinet/ip.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>

#endif


#include <MultiSense/src/tools/Logger.h>
#include <MultiSense/src/tools/Utils.h>
#include "MultiSense/src/crl_camera/CRLVirtualCamera.h"
#include "MultiSense/src/crl_camera/CRLPhysicalCamera.h"
#include <unistd.h>


CameraConnection::CameraConnection() {

}

void CameraConnection::updateActiveDevice(AR::Element *dev) {

    if (!dev->parameters.initialized) {
        auto *p = &dev->parameters;

        auto conf = camPtr->getCameraInfo().imgConf;


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

    if (dev->parameters.update) {
        auto *p = &dev->parameters;

        camPtr->setExposureParams(p->ep);
        camPtr->setWhiteBalance(p->wb);
        camPtr->setPostFilterStrength(p->stereoPostFilterStrength);
        camPtr->setGamma(p->gamma);
        camPtr->setFps(p->fps);
        camPtr->setGain(p->gain);

        auto conf = camPtr->getCameraInfo().imgConf;


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

        dev->parameters.update = false;
    }

    // Set the correct resolution. Will only update if changed.
    camPtr->setResolution(dev->selectedMode);

    // Handle streams enabling/disable
    // Enable sources that are in userRequested but not in enabled
    for (const auto &s: dev->userRequestedSources) {
        if (!Utils::isInVector(dev->enabledStreams, s)) {
            // Enable stream and push back if it is successfully enabled
            if (s == "None")
                continue;

            if (camPtr->start(dev->selectedMode, s)) {
                dev->enabledStreams.push_back(s);
            }
            else
                Log::Logger::getInstance()->info("Failed to enabled stream {}", s);
        }
    }

    // Disable sources that are in enabled but not in userRequested
    for (const auto &s: dev->enabledStreams) {
        if (!Utils::isInVector(dev->userRequestedSources, s)) {
            // Enable stream and push back if it is successfully enabled
            if (camPtr->stop(s))
                Utils::removeFromVector(&dev->enabledStreams, s);
            else
                Log::Logger::getInstance()->info("Failed to disable stream {}", s);
        }
    }

}

void CameraConnection::onUIUpdate(std::vector<AR::Element> *pVector) {
    // If no device is connected then return
    if (pVector == nullptr)
        return;
    // Check for actions on each element
    for (auto &dev: *pVector) {

        if (dev.state == AR_STATE_RESET) {
            disableCrlCamera(dev);
            dev.state = AR_STATE_UNAVAILABLE;
            for (auto &s: dev.streams) {
                s.second.playbackStatus = AR_PREVIEW_NONE;
            }
            return;
        }



        // Connect if we click a device or if it is just added
        if ((dev.clicked && dev.state != AR_STATE_ACTIVE) || dev.state == AR_STATE_JUST_ADDED) {

            // reset other active device if present. So loop over all devices again. quick hack is to updaet state for the newly connect device/clicked device ot just added to enter this if statement on next render loop
            bool resetOtherDeviceFirst = false;
            for (auto &d: *pVector) {
                if (d.state == AR_STATE_ACTIVE && d.name != dev.name) {
                    d.state = AR_STATE_RESET;
                    Log::Logger::getInstance()->info("Call to reset state requested for profile {}", d.name);
                    resetOtherDeviceFirst = true;
                }
            }

            if (!resetOtherDeviceFirst)
                connectCrlCamera(dev);
            else
                dev.state = AR_STATE_JUST_ADDED;

            break;
        }

        updateDeviceState(&dev);

        // Make sure inactive devices' preview are not drawn.
        if (dev.state != AR_STATE_ACTIVE) {

            continue;
        }


        updateActiveDevice(&dev);

        // Disable if we click a device already connected
        if (dev.clicked && dev.state == AR_STATE_ACTIVE) {
            // Disable all streams and delete camPtr on next update
            Log::Logger::getInstance()->info("Call to reset state requested for profile {}", dev.name);
            dev.state = AR_STATE_RESET;
        }
    }


}

void CameraConnection::connectCrlCamera(AR::Element &dev) {
    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    bool connected{};

    Log::Logger::getInstance()->info("Connect.");


    if (dev.cameraName == "Virtual Camera") {
        camPtr = std::make_shared<CRLVirtualCamera>();
        connected = camPtr->connect("None");
        if (connected) {
            dev.state = AR_STATE_ACTIVE;
            dev.cameraName = "Virtual Camera";
            dev.IP = "Local";
            lastActiveDevice = dev.name;


            dev.modes.emplace_back("1920x1080");
            dev.sources.emplace_back("rowbot_short.mpg");
            dev.selectedMode = Utils::stringToCameraResolution(dev.modes.front());


            /*
            AR::StreamingModes virtualCam{};
            virtualCam.name = "1. Virtual Left";
            virtualCam.sources.emplace_back("rowbot_short.mpg");
            virtualCam.sources.emplace_back("crl_jeep_multisense_front_left_image_rect.mp4");
            virtualCam.sources.emplace_back("jeep_disparity.mp4");

            virtualCam.streamIndex = AR_PREVIEW_VIRTUAL_LEFT;
            std::string modeName = "1920x1080";
            virtualCam.modes.emplace_back(modeName);
            virtualCam.selectedStreamingMode = Utils::stringToCameraResolution(virtualCam.modes.front());
            virtualCam.selectedStreamingSource = virtualCam.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_LEFT] = virtualCam;

            AR::StreamingModes virtualRight{};
            virtualRight.name = "2. Virtual Right";
            virtualRight.sources.emplace_back("rowbot_short.mpg");
            virtualRight.sources.emplace_back("crl_jeep_multisense_front_left_image_rect.mp4");
            virtualRight.streamIndex = AR_PREVIEW_VIRTUAL_RIGHT;
            modeName = "1920x1080";
            virtualRight.modes.emplace_back(modeName);
            virtualRight.selectedStreamingMode = Utils::stringToCameraResolution(virtualRight.modes.front());
            virtualRight.selectedStreamingSource = virtualRight.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_RIGHT] = virtualRight;
            AR::StreamingModes aux{};
            aux.name = "3. Virtual Auxiliary";
            aux.sources.emplace_back("rowbot_short.mpg");
            aux.sources.emplace_back("jeep_multisense_front_aux_image_color.mp4");
            aux.streamIndex = AR_PREVIEW_VIRTUAL_AUX;
            modeName = "1920x1080";
            aux.modes.emplace_back(modeName);
            aux.selectedStreamingMode = Utils::stringToCameraResolution(aux.modes.front());
            aux.selectedStreamingSource = aux.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_AUX] = aux;

            AR::StreamingModes virutalPC{};
            virutalPC.name = "4. Virtual Point cloud";
            virutalPC.sources.emplace_back("depth");
            virutalPC.streamIndex = AR_PREVIEW_VIRTUAL_POINT_CLOUD;
            modeName = "1920x1080";
            virutalPC.modes.emplace_back(modeName);
            virutalPC.selectedStreamingMode = Utils::stringToCameraResolution(virutalPC.modes.front());
            virutalPC.selectedStreamingSource = virutalPC.sources.front();
            dev.streams[AR_PREVIEW_VIRTUAL_POINT_CLOUD] = virutalPC;
*/

            Log::Logger::getInstance()->info("Creating new Virtual Camera.");


        } else
            dev.state = AR_STATE_UNAVAILABLE;

    } else {

        if (!setNetworkAdapterParameters(dev)) {
            dev.state = AR_STATE_UNAVAILABLE;
            return;
        }

        Log::Logger::getInstance()->info("Creating new physical camera.");

        camPtr = std::make_shared<CRLPhysicalCamera>();
        connected = camPtr->connect(dev.IP);
        if (connected) {
            dev.state = AR_STATE_ACTIVE;
            dev.cameraName = camPtr->getCameraInfo().devInfo.name;
            setStreamingModes(dev);
            lastActiveDevice = dev.name;

        } else {
            disableCrlCamera(dev);

        }
    }
}

void CameraConnection::setStreamingModes(AR::Element &dev) {

    auto supportedModes = camPtr->getCameraInfo().supportedDeviceModes;

    dev.modes.clear();
    dev.sources.clear();
    dev.sources.emplace_back("None");

    initCameraModes(&dev.modes, supportedModes);
    filterAvailableSources(&dev.sources, maskArrayAll);
    dev.selectedSourceIndex = 0;
    dev.selectedMode = CRL_RESOLUTION_960_600_256;

    if (dev.modes.empty() || dev.sources.empty()){
        Log::Logger::getInstance()->info("Modes and Sources empty for physical camera");
    }

    /*
    AR::StreamingModes left{};
    left.name = "1. Left Sensor";
    left.streamIndex = AR_PREVIEW_LEFT;
    initCameraModes(&left.modes, supportedModes);
    filterAvailableSources(&left.sources, maskArrayLeft);
    left.selectedStreamingMode = Utils::stringToCameraResolution(left.modes.front());
    left.selectedStreamingSource = left.sources.front();


    AR::StreamingModes right{};
    right.name = "2. Right Sensor";
    right.streamIndex = AR_PREVIEW_RIGHT;
    initCameraModes(&right.modes, supportedModes);
    filterAvailableSources(&right.sources, maskArrayRight);
    right.selectedStreamingMode = Utils::stringToCameraResolution(right.modes.front());
    right.selectedStreamingSource = right.sources.front();

    AR::StreamingModes disparity{};
    disparity.name = "3. Disparity Image";
    disparity.streamIndex = AR_PREVIEW_DISPARITY;
    initCameraModes(&disparity.modes, supportedModes);
    filterAvailableSources(&disparity.sources, maskArrayDisparity);
    disparity.selectedStreamingMode = Utils::stringToCameraResolution(disparity.modes.front());
    disparity.selectedStreamingSource = disparity.sources.front();

    AR::StreamingModes aux{};
    aux.name = "4. Aux Sensor";
    aux.streamIndex = AR_PREVIEW_AUXILIARY;
    initCameraModes(&aux.modes, supportedModes);
    filterAvailableSources(&aux.sources, maskArrayAux);
    aux.selectedStreamingMode = Utils::stringToCameraResolution(aux.modes.front());
    aux.selectedStreamingSource = aux.sources.front();

    AR::StreamingModes pointCloud{};
    pointCloud.name = "5. Point Cloud";
    pointCloud.streamIndex = AR_PREVIEW_POINT_CLOUD;
    initCameraModes(&pointCloud.modes, supportedModes);
    filterAvailableSources(&pointCloud.sources, maskArrayDisparity);
    pointCloud.selectedStreamingMode = Utils::stringToCameraResolution(pointCloud.modes.front());
    pointCloud.selectedStreamingSource = pointCloud.sources.front();


    dev.streams[AR_PREVIEW_LEFT] = left;
    dev.streams[AR_PREVIEW_RIGHT] = right;
    dev.streams[AR_PREVIEW_DISPARITY] = disparity;
    dev.streams[AR_PREVIEW_AUXILIARY] = aux;
    dev.streams[AR_PREVIEW_POINT_CLOUD] = pointCloud;
    */

    Log::Logger::getInstance()->info("setting available streaming modes");


}

void CameraConnection::initCameraModes(std::vector<std::string> *modes,
                                       std::vector<crl::multisense::system::DeviceMode> deviceModes) {
    for (auto mode: deviceModes) {
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
    }
}


bool CameraConnection::setNetworkAdapterParameters(AR::Element &dev) {

    hostAddress = dev.IP;

    // TODO recheck if we want to start using exceptions for stuff or if this is fine
    try {
        std::string last_element(hostAddress.substr(hostAddress.rfind('.')));
        hostAddress.replace(hostAddress.rfind('.'), last_element.length(), ".2");
    }
    catch (std::out_of_range &exception) {
        Log::Logger::getInstance()->error(
                "Trying to configure adapter '{}' with source IP: '{}', but address does not seem like an ipv4 address",
                dev.interfaceName, dev.IP);
        Log::Logger::getInstance()->error("Exception message: '{}'", exception.what());
        return false;
    }

#ifdef WIN32

    /* Variables where handles to the added IP are returned */
    ULONG NTEInstance = 0;
    // Attempt to connect to camera and post some info


    LPVOID lpMsgBuf;

    unsigned long ulAddr = inet_addr(hostAddress.c_str());
    unsigned long ulMask = inet_addr("255.255.255.0");
    if ((dwRetVal = AddIPAddress(ulAddr,
        ulMask,
        dev.interfaceIndex,
        &NTEContext, &NTEInstance)) == NO_ERROR) {
        printf("\tIPv4 address %s was successfully added.\n\n", hostAddress.c_str());
    }
    else {
        printf("AddIPAddress failed with error: %d\n", dwRetVal);
        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, dwRetVal, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),       // Default language
            (LPTSTR)&lpMsgBuf, 0, NULL)) {
            printf("\tError: %s", lpMsgBuf);
            LocalFree(lpMsgBuf);
        }

    }


#else
    /** SET NETWORK PARAMETERS FOR THE ADAPTER */
    if ((sd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        fprintf(stderr, "Error creating socket: %s\n", strerror(errno));
        Log::Logger::getInstance()->error("Error in creating socket to configure network adapter: '{}'",
                                          strerror(errno));

        return false;
    }
    // Specify interface name
    const char *interface = dev.interfaceName.c_str();
    if (setsockopt(sd, SOL_SOCKET, SO_BINDTODEVICE, interface, 15) < 0) {
        Log::Logger::getInstance()->error("Could not bind socket to adapter {}, '{}'", dev.interfaceName,
                                          strerror(errno));
    };

    struct ifreq ifr{};
    /// note: no pointer here
    struct sockaddr_in inet_addr{}, subnet_mask{};
    /* get interface name */
    /* Prepare the struct ifreq */
    bzero(ifr.ifr_name, IFNAMSIZ);
    strncpy(ifr.ifr_name, interface, IFNAMSIZ);

    /// note: prepare the two struct sockaddr_in
    inet_addr.sin_family = AF_INET;
    int inet_addr_config_result = inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

    subnet_mask.sin_family = AF_INET;
    int subnet_mask_config_result = inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));

    /* Call ioctl to configure network devices */
    /// put addr in ifr structure
    memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
    int ioctl_result = ioctl(sd, SIOCSIFADDR, &ifr);  // Set IP address
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFADDR: %s", strerror(errno));
        Log::Logger::getInstance()->error("Could not set ip address on {}, reason: {}", dev.interfaceName,
                                          strerror(errno));

    }

    /// put mask in ifr structure
    memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
    ioctl_result = ioctl(sd, SIOCSIFNETMASK, &ifr);   // Set subnet mask
    if (ioctl_result < 0) {
        fprintf(stderr, "ioctl SIOCSIFNETMASK: %s", strerror(errno));
        Log::Logger::getInstance()->error("Could not set subnet mask address on {}, reason: {}", dev.interfaceName,
                                          strerror(errno));
    }

    strncpy(ifr.ifr_name, interface, sizeof(ifr.ifr_name));//interface name where you want to set the MTU
    ifr.ifr_mtu = 7200; //your MTU size here
    if (ioctl(sd, SIOCSIFMTU, (caddr_t) &ifr) < 0) {
        Log::Logger::getInstance()->error("Failed to set mtu size {} on adapter {}", 7200,
                                          dev.interfaceName.c_str());
    } else {
        Log::Logger::getInstance()->error("Set Mtu size to {} on adapter {}", 7200,
                                          dev.interfaceName.c_str());
    }


#endif
    return true;
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
    lastActiveDevice = "-1";

    Log::Logger::getInstance()->info("Disconnecting profile {} using camera {}", dev.name.c_str(),
                                     dev.cameraName.c_str());

    camPtr.reset();
    dev.userRequestedSources.clear();
    dev.enabledStreams.clear();
    dev.selectedSourceMap.clear();
    dev.selectedSourceIndexMap.clear();

    /*
#ifdef WIN32

    if ((dwRetVal = DeleteIPAddress(NTEContext)) == NO_ERROR) {
        printf("\tIPv4 address %s was successfully deleted.\n", hostAddress.c_str());
    }
    else {
        printf("\tDeleteIPAddress failed with error: %d\n", dwRetVal);
        LPVOID lpMsgBuf;

        if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, dwRetVal, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),       // Default language
            (LPTSTR)&lpMsgBuf, 0, NULL)) {
            printf("\tError: %s", lpMsgBuf);
            LocalFree(lpMsgBuf);
        }
    }
#endif
*/
}

CameraConnection::~CameraConnection() {
    // Make sure delete the camPtr for physical cameras so we run destructor on the physical camera class which
    // stops all streams on the camera

#ifndef WIN32
    if (sd != -1)
        close(sd);
#endif // !WIN32


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