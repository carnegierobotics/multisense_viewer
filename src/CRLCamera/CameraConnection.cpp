//
// Created by magnus on 3/21/22.
//

#include "CameraConnection.h"


#ifdef WIN32
//#define _WINSOCKAPI_    // stops windows.h including winsock.h

#ifndef _WINSOCKAPI_
#include <WinSock2.h>
#include <WS2tcpip.h>
#endif

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
#include <unistd.h>

#endif


#include <MultiSense/src/Tools/Logger.h>
#include <MultiSense/src/Tools/Utils.h>
#include "MultiSense/src/CRLCamera/CRLVirtualCamera.h"
#include "MultiSense/src/CRLCamera/CRLPhysicalCamera.h"


void CameraConnection::updateActiveDevice(MultiSense::Device *dev) {
    auto *p = &dev->parameters;
    if (!dev->parameters.initialized) {
        const auto &conf = camPtr->getCameraInfo(0).imgConf;
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

        const auto &lightConf = camPtr->getCameraInfo(0).lightConf;
        p->light.numLightPulses = lightConf.getNumberOfPulses();
        p->light.dutyCycle = lightConf.getDutyCycle(0);
        p->light.flashing = lightConf.getFlash();
        p->light.startupTime = lightConf.getStartupTime();

        p->stereoPostFilterStrength = conf.stereoPostFilterStrength();
        p->initialized = true;
    }

    for (auto &ch: dev->channelInfo) {
        if (ch.state != AR_STATE_ACTIVE)
            continue;

        if (dev->parameters.update)
            pool->Push(CameraConnection::setAdditionalParametersTask, this, p->fps, p->gain, p->gamma, ch.index,
                       p->stereoPostFilterStrength, dev);
    }

    if (dev->parameters.ep.update)
        pool->Push(CameraConnection::setExposureTask, this, &p->ep, dev);

    if (dev->parameters.wb.update)
        pool->Push(CameraConnection::setWhiteBalanceTask, this, &p->wb, dev);

    if (dev->parameters.light.update)
        pool->Push(CameraConnection::setLightingTask, this, &p->light, dev);
    // start streams
    /*
    for (int i = 0; i < dev->win.size(); ++i) {
        MultiSense::PreviewWindow win = dev->win[i];
        MultiSense::ChannelInfo& chInfo = dev->channelInfo[win.selectedRemoteHeadIndex];

        if (!Utils::isInVector(chInfo.enabledStreams, )){
            pool->Push(CameraConnection::startStreamTaskRemoteHead, this, dev, win.selectedSource, win.selectedRemoteHeadIndex);
            dev->channelInfo[win.selectedRemoteHeadIndex].enabledStreams.emplace_back(win.selectedSource);
        }
    }
     */
    // Set the correct resolution. Will only update if changed.
    for (auto &ch: dev->channelInfo) {
        if (ch.state == AR_STATE_ACTIVE) {
            //pool->Push(CameraConnection::setResolutionTask, this, ch.selectedMode, ch.index);
            setResolutionTask(this, ch.selectedMode, ch.index);
        }

    }

    // Start requested streams
    for (auto &ch: dev->channelInfo) {
        if (ch.state != AR_STATE_ACTIVE)
            continue;
        for (const auto &requested: ch.requestedStreams) {
            if (!Utils::isInVector(ch.enabledStreams, requested)) {
                pool->Push(CameraConnection::startStreamTaskRemoteHead, this, dev, requested, ch.index);
                ch.enabledStreams.emplace_back(requested);
            }
        }
    }

    // Stop if the stream is no longer requested
    for (auto &ch: dev->channelInfo) {
        if (ch.state != AR_STATE_ACTIVE)
            continue;
        for (const auto &enabled: ch.enabledStreams) {
            if (!Utils::isInVector(ch.requestedStreams, enabled)) {
                pool->Push(CameraConnection::stopStreamTaskRemoteHead, this, dev, enabled, ch.index);
                Utils::removeFromVector(&ch.enabledStreams, enabled);
            }
        }
    }

    // Stop streams

    /*
    // Handle streams enabling/disable
    // Enable sources that are in userRequested but not in enabled
    for (const auto &ch: dev->selectedRemoteHeadIndexMap) {
        for (const auto &s: dev->userRequestedSourcesMap[ch.second]) {
            if (!Utils::isInVector(dev->enabledStreams[ch.second], s)) {
                // Enable stream and push back if it is successfully enabled
                if (s == "Source")
                    continue;
                if (pool->getTaskListSize() > 0)
                    continue;

                if (dev->baseCameraUnit == CRL_BASE_REMOTE_HEAD)
                    pool->Push(CameraConnection::startStreamTaskRemoteHead, this, dev, s, dev->selectedRemoteHeadIndex);
                else
                    pool->Push(CameraConnection::startStreamTask, this, dev, s);
            }
        }

        // Disable sources that are in enabled but not in userRequested
        for (const auto &s: dev->enabledStreams[ch.second]) {
            if (!Utils::isInVector(dev->userRequestedSourcesMap[ch.second], s)) {
                // Enable stream and push back if it is successfully enabled
                if (pool->getTaskListSize() > 0)
                    continue;

                if (dev->baseCameraUnit == CRL_BASE_REMOTE_HEAD)
                    pool->Push(CameraConnection::stopStreamTaskRemoteHead, this, dev, s, dev->selectedRemoteHeadIndex);
                else
                    pool->Push(CameraConnection::stopStreamTask, this, dev, s);

            }
        }
    }
     */
}

void CameraConnection::onUIUpdate(std::vector<MultiSense::Device> *pVector, bool shouldConfigNetwork, bool isRemoteHead) {
    // If no device is connected then return
    if (pVector == nullptr)
        return;
    // Check for actions on each element
    for (auto &dev: *pVector) {

        if (dev.state == AR_STATE_RESET || dev.state == AR_STATE_DISCONNECT_AND_FORGET) {
            Log::Logger::getInstance()->info("Pushing disconnect task");
            // Only call this task if it is not already running
            if (!processingDisconnectTask) {
                pool->Push(CameraConnection::disconnectCRLCameraTask, this, &dev);
                processingDisconnectTask = true;
            }
            return;
        }

        // Connect if we click a device or if it is just added
        if ((dev.clicked && dev.state != AR_STATE_ACTIVE) || dev.state == AR_STATE_JUST_ADDED) {
            // reset other active device if present. So loop over all devices again. quick hack is to updaet state for the newly connect device/clicked device ot just added to enter this if statement on next render loop
            bool resetOtherDeviceFirst = false;
            MultiSense::Device *otherDev;
            for (auto &d: *pVector) {
                if (d.state == AR_STATE_ACTIVE && d.name != dev.name) {
                    d.state = AR_STATE_RESET;
                    Log::Logger::getInstance()->info("Call to reset state requested for profile {}", d.name);
                    resetOtherDeviceFirst = true;
                    otherDev = &d;
                }
            }
            if (resetOtherDeviceFirst) {
                pool->Push(CameraConnection::disconnectCRLCameraTask, this, otherDev);
            }
            setNetworkAdapterParameters(dev, shouldConfigNetwork);
            pool->Push(CameraConnection::connectCRLCameraTask, this, &dev, isRemoteHead);
            dev.state = AR_STATE_CONNECTING;
            break;
        }

        updateDeviceState(&dev);
        // Rest of this function is only for active devices
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

void CameraConnection::setStreamingModes(MultiSense::Device &dev) {
    // Find all possible streaming modes
    dev.channelInfo.resize(4); // max number of remote heads
    for (auto ch: dev.channelConnections) {
        MultiSense::ChannelInfo chInfo;
        chInfo.availableSources.clear();
        chInfo.modes.clear();
        chInfo.availableSources.emplace_back("Source");
        chInfo.index = ch;
        chInfo.state = AR_STATE_ACTIVE;

        filterAvailableSources(&chInfo.availableSources, maskArrayAll, ch);
        const auto &supportedModes = camPtr->getCameraInfo(ch).supportedDeviceModes;
        initCameraModes(&chInfo.modes, supportedModes);
        chInfo.selectedMode = CRL_RESOLUTION_NONE;



        dev.channelInfo.at(ch) = chInfo;
        // Check for previous user profiles attached to this hardware
    }
    /*
    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile("crl.ini");
    if (rc < 0) { /* handle error  }
    else {
        // A serial number is the section identifier of a profile. I can have three following states
        // - Not Exist
        // - Exist
        std::string cameraSerialNumber = camPtr->getCameraInfo().devInfo.serialNumber;
        if (ini.SectionExists(cameraSerialNumber.c_str())) {
            std::string profileName = ini.GetValue(cameraSerialNumber.c_str(), "ProfileName", "default");
            std::string mode = ini.GetValue(cameraSerialNumber.c_str(), "Mode", "default");
            std::string layout = ini.GetValue(cameraSerialNumber.c_str(), "Layout", "default");
            std::string enabledSources = ini.GetValue(cameraSerialNumber.c_str(), "EnabledSources", "default");
            std::string previewOne = ini.GetValue(cameraSerialNumber.c_str(), "Preview1", "Source");
            std::string previewTwo = ini.GetValue(cameraSerialNumber.c_str(), "Preview2", "Source");
            Log::Logger::getInstance()->info("Using Layout {} and camera resolution {}", layout, mode);

            dev.layout = static_cast<PreviewLayout>(std::stoi(layout));
            dev.selectedMode = static_cast<CRLCameraResolution>(std::stoi(mode));
            dev.selectedModeIndex = std::stoi(mode);

            // Create previews
            for (int i = 0; i < AR_PREVIEW_TOTAL_MODES; ++i) {
                std::string prev = "Preview" + std::to_string(i + 1);
                std::string source = std::string(ini.GetValue(cameraSerialNumber.c_str(), prev.c_str(), ""));
                if (!source.empty()) {
                    Log::Logger::getInstance()->info(".ini file: Starting source '{}' for preview {}", source,
                                                     i + 1);
                    dev.selectedSourceMap[i] = source;
                    //dev.userRequestedSources.emplace_back(source);

                    auto it = find(dev.sources.begin(), dev.sources.end(), source);
                    if (it != dev.sources.end()) {
                        dev.selectedSourceIndexMap[i] = static_cast<uint32_t>(it - dev.sources.begin());
                    }
                }
            }
        }
    }
    */

    for (int i = 0; i < AR_PREVIEW_TOTAL_MODES; ++i) {
        dev.win[i].availableRemoteHeads.clear();
        for (auto ch: dev.channelConnections) {
            dev.win[i].availableRemoteHeads.push_back(std::to_string(ch));
        }
    }
    // Populate streaming info for preview info
    Log::Logger::getInstance()->info("setting available streaming modes");
}

void CameraConnection::initCameraModes(std::vector<std::string> *modes,
                                       std::vector<crl::multisense::system::DeviceMode> deviceModes) {
    for (const auto &mode: deviceModes) {
        std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                               std::to_string(mode.disparities) + "x";
        modes->emplace_back(modeName);
    }

}

void CameraConnection::filterAvailableSources(std::vector<std::string> *sources, std::vector<uint32_t> maskVec,
                                              uint32_t idx) {
    uint32_t bits = camPtr->getCameraInfo(idx).supportedSources;
    for (auto mask: maskVec) {
        bool enabled = (bits & mask);
        if (enabled) {
            sources->emplace_back(Utils::dataSourceToString(mask));
        }
    }
}


bool CameraConnection::setNetworkAdapterParameters(MultiSense::Device &dev, bool shouldConfigNetwork) {

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
    if (shouldConfigNetwork) {

#ifdef WIN32


        ///* Variables where handles to the added IP are returned */
        //ULONG NTEInstance = 0;
        //// Attempt to connect to camera and post some info


        //LPVOID lpMsgBuf = nullptr;

        //unsigned long ulAddr = inet_addr(hostAddress.c_str());
        //unsigned long ulMask = inet_addr("255.255.255.0");
        //if ((dwRetVal = AddIPAddress(ulAddr,
        //    ulMask,
        //    dev.interfaceIndex,
        //    &NTEContext, &NTEInstance)) == NO_ERROR) {
        //    printf("\tIPv4 address %s was successfully added.\n\n", hostAddress.c_str());
        //}
        //else {
        //    printf("AddIPAddress failed with error: %d\n", dwRetVal);
        //    if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        //        NULL, dwRetVal, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),       // Default language
        //        (LPTSTR)&lpMsgBuf, 0, NULL)) {
        //        printf("\tError: %p", std::addressof(lpMsgBuf));
        //        LocalFree(lpMsgBuf);
        //    }

        //}


#else
        /** SET NETWORK PARAMETERS FOR THE ADAPTER **/
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
    }

#endif
    return true;
}

void CameraConnection::updateDeviceState(MultiSense::Device *dev) {

}

void CameraConnection::addIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value) {
    int ret = ini->SetValue(section.c_str(), key.c_str(), value.c_str());

    if (ret < 0)
        Log::Logger::getInstance()->error("Section: {} Updated {} to {}", section, key, value);
    else
        Log::Logger::getInstance()->info("Section: {} Updated {} to {}", section, key, value);

}

void CameraConnection::deleteIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value) {
    int ret = ini->DeleteValue(section.c_str(), key.c_str(), value.c_str());

    if (ret < 0)
        Log::Logger::getInstance()->error("Section: {} deleted {}", section, key, value);
    else
        Log::Logger::getInstance()->info("Section: {} deleted {}", section, key, value);

}

CameraConnection::~CameraConnection() {
    // Make sure delete the camPtr for physical cameras so we run destructor on the physical camera class which
    // stops all streams on the camera
    camPtr.reset();
#ifndef WIN32
    if (sd != -1)
        close(sd);
#endif // !WIN32
}


void CameraConnection::connectCRLCameraTask(void *context, MultiSense::Device *dev, bool isRemoteHead) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    Log::Logger::getInstance()->info("Connect.");
    Log::Logger::getInstance()->info("Creating new physical camera.");
    app->camPtr = std::make_unique<CRLPhysicalCamera>();
    // If we successfully connect
    dev->channelConnections = app->camPtr->connect(dev->IP, isRemoteHead);

    if (!dev->channelConnections.empty()) {
        dev->state = AR_STATE_ACTIVE;
        dev->cameraName = app->camPtr->getCameraInfo(0).devInfo.name;
        dev->serialName = app->camPtr->getCameraInfo(0).devInfo.serialNumber;
        app->setStreamingModes(*dev); // TODO Race condition in altering the *dev piece of memory
        app->lastActiveDevice = dev->name;
    } else {
        dev->state = AR_STATE_UNAVAILABLE;
        app->lastActiveDevice = "-1";
    }


}

void CameraConnection::disconnectCRLCameraTask(void *context, MultiSense::Device *dev) {
    auto *app = reinterpret_cast<CameraConnection *>(context);


    Log::Logger::getInstance()->info("Disconnecting profile {} using camera {}", dev->name.c_str(),
                                     dev->cameraName.c_str());
    // Save settings to file. Attempt to create a new file if it doesn't exist
    /*
    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile("crl.ini");
    if (rc < 0) {
        // File doesn't exist error, then create one
        if (rc == SI_FILE && errno == ENOENT) {
            std::ofstream output("crl.ini");
            rc = ini.LoadFile("crl.ini");
        } else
            Log::Logger::getInstance()->error("Failed to create profile configuration file\n");
    }

    std::string CRLSerialNumber = dev->serialName;
    // If sidebar is empty or we dont recognize any serial numbers in the crl.ini file then clear it.

    // new entry given we have a valid ini file entry
    if (rc >= 0 && !CRLSerialNumber.empty()) {
        // Profile Data
        addIniEntry(&ini, CRLSerialNumber, "ProfileName", dev->name);
        addIniEntry(&ini, CRLSerialNumber, "AdapterName", dev->interfaceName);
        addIniEntry(&ini, CRLSerialNumber, "CameraName", dev->cameraName);
        addIniEntry(&ini, CRLSerialNumber, "IP", dev->IP);
        addIniEntry(&ini, CRLSerialNumber, "AdapterIndex", std::to_string(dev->interfaceIndex));
        addIniEntry(&ini, CRLSerialNumber, "State", std::to_string((int) dev->state));
        // Preview Data
        addIniEntry(&ini, CRLSerialNumber, "Mode", std::to_string((int) dev->selectedMode));
        addIniEntry(&ini, CRLSerialNumber, "Layout", std::to_string((int) dev->layout));
        for (int i = 0; i < AR_PREVIEW_TOTAL_MODES; ++i) {
            if (dev->selectedSourceMap.contains(i)) {
                std::string key = "Preview" + std::to_string(i + 1);
                addIniEntry(&ini, CRLSerialNumber, key, dev->selectedSourceMap[i]);
            }
        }
    }

    // delete entry if we gave the disconnect and reset flag Otherwise just normal disconnect


    // save the data back to the file
    rc = ini.SaveFile("crl.ini");
    if (rc < 0) {
        Log::Logger::getInstance()->info("Failed to save crl.ini file. Err: {}", rc);

    }


    app->lastActiveDevice = "-1";
    dev->userRequestedSourcesMap.clear();
    dev->enabledStreams.clear();
    dev->selectedSourceMap.clear();
    dev->selectedSourceIndexMap.clear();
       */
    if (dev->state == AR_STATE_DISCONNECT_AND_FORGET) {
        //int done = ini.Delete(CRLSerialNumber.c_str(), nullptr);
        dev->state = AR_STATE_REMOVE_FROM_LIST;
        //Log::Logger::getInstance()->info("Deleted saved profile for serial: {}", CRLSerialNumber);

    } else {
        dev->state = AR_STATE_DISCONNECTED;
    }    app->processingDisconnectTask = false;

}

void CameraConnection::setExposureTask(void *context, void *arg1, MultiSense::Device *dev) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    auto *ep = reinterpret_cast<ExposureParams *>(arg1);

    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setExposureParams(*ep);
    app->updateFromCameraParameters(dev, 0);
}

void CameraConnection::setWhiteBalanceTask(void *context, void *arg1, MultiSense::Device *dev) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    auto *ep = reinterpret_cast<WhiteBalanceParams *>(arg1);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setWhiteBalance(*ep);
    app->updateFromCameraParameters(dev, 0);
}

void CameraConnection::setAdditionalParametersTask(void *context, float fps, float gain, float gamma, float spfs,
                                                   uint32_t index,
                                                   MultiSense::Device *dev) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    //app->camPtr->setGamma(gamma);
    //app->camPtr->setGain(gain);
    app->camPtr->setFps(fps, index);
    //app->camPtr->setPostFilterStrength(spfs);
    app->updateFromCameraParameters(dev, index);

}

void CameraConnection::setLightingTask(void *context, void *arg1, MultiSense::Device *dev) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    auto *light = reinterpret_cast<LightingParams *>(arg1);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setLighting(*light);
    app->updateFromCameraParameters(dev, 0);
    // TODO implement
}

void CameraConnection::updateFromCameraParameters(MultiSense::Device *dev, uint32_t index) const {
    // Query the camera for new values and update the GUI. It is a way to see if the actual value was set.
    const auto &conf = camPtr->getCameraInfo(0).imgConf;
    auto *p = &dev->parameters;
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
    dev->parameters.update = false;

}

void CameraConnection::startStreamTaskRemoteHead(void *context, MultiSense::Device *dev, std::string src,
                                            uint32_t remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);

    if (app->camPtr->start(src, remoteHeadIndex));
    else
        Log::Logger::getInstance()->info("Failed to enabled stream {}", src);


}

void CameraConnection::stopStreamTaskRemoteHead(void *context, MultiSense::Device *dev, std::string src,
                                           uint32_t remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);

    if (app->camPtr->stop(src, remoteHeadIndex));
    else
        Log::Logger::getInstance()->info("Failed to disable stream {}", src);


}


void CameraConnection::setResolutionTask(void *context, CRLCameraResolution arg1, uint32_t idx) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setResolution(arg1, idx);

}

