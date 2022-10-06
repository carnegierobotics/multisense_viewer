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
#include <WinRegEditor.h>

#else

#include <net/if.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <unistd.h>

#endif

#include <MultiSense/src/Tools/Utils.h>
#include "MultiSense/external/simpleini/SimpleIni.h"


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
            pool->Push(CameraConnection::setAdditionalParametersTask, this, p->fps, p->gain, p->gamma,
                       p->stereoPostFilterStrength, p->hdrEnabled, ch.index, dev);
    }

    if (dev->parameters.ep.update)
        pool->Push(CameraConnection::setExposureTask, this, &p->ep, dev);

    if (dev->parameters.wb.update)
        pool->Push(CameraConnection::setWhiteBalanceTask, this, &p->wb, dev);

    if (dev->parameters.light.update)
        pool->Push(CameraConnection::setLightingTask, this, &p->light, dev);
    // Set the correct resolution. Will only update if changed.
    for (auto &ch: dev->channelInfo) {
        if (ch.state == AR_STATE_ACTIVE) {
            pool->Push(CameraConnection::setResolutionTask, this, ch.selectedMode, ch.index);
            //setResolutionTask(this, ch.selectedMode, ch.index);
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

}

void
CameraConnection::onUIUpdate(std::vector<MultiSense::Device> *pVector, bool shouldConfigNetwork, bool isRemoteHead) {
    // If no device is connected then return
    if (pVector == nullptr)
        return;
    // Check for actions on each element
    for (auto &dev: *pVector) {

        if (dev.state == AR_STATE_RESET || dev.state == AR_STATE_DISCONNECT_AND_FORGET) {
            // Only call this task if it is not already running
            saveProfileAndDisconnect(&dev);
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
                saveProfileAndDisconnect(otherDev);
            }
            pool->Push(CameraConnection::connectCRLCameraTask, this, &dev, isRemoteHead, shouldConfigNetwork);
            dev.state = AR_STATE_CONNECTING;
            break;
        }
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

        // Check for previous user profiles attached to this hardware
        for (int i = 0; i < AR_PREVIEW_TOTAL_MODES; ++i) {
            dev.win[i].availableRemoteHeads.push_back(std::to_string(ch));
        }

        CSimpleIniA ini;
        ini.SetUnicode();
        SI_Error rc = ini.LoadFile("crl.ini");
        if (rc < 0) {} // Handle error
        else {
            // A serial number is the section identifier of a profile. I can have three following states
            // - Not Exist
            // - Exist
            std::string cameraSerialNumber = camPtr->getCameraInfo(ch).devInfo.serialNumber;
            if (ini.SectionExists(cameraSerialNumber.c_str())) {
                std::string profileName = ini.GetValue(cameraSerialNumber.c_str(), "ProfileName", "default");
                std::string mode = ini.GetValue(cameraSerialNumber.c_str(),
                                                std::string("Mode" + std::to_string(ch)).c_str(), "default");
                std::string layout = ini.GetValue(cameraSerialNumber.c_str(), "Layout", "default");
                std::string enabledSources = ini.GetValue(cameraSerialNumber.c_str(), "EnabledSources", "default");
                std::string previewOne = ini.GetValue(cameraSerialNumber.c_str(), "Preview1", "Source");
                std::string previewTwo = ini.GetValue(cameraSerialNumber.c_str(), "Preview2", "Source");
                Log::Logger::getInstance()->info("Using Layout {} and camera resolution {}", layout, mode);

                dev.layout = static_cast<PreviewLayout>(std::stoi(layout));
                chInfo.selectedMode = static_cast<CRLCameraResolution>(std::stoi(mode));
                chInfo.selectedModeIndex = std::stoi(mode);
                // Create previews
                for (int i = 0; i < AR_PREVIEW_TOTAL_MODES; ++i) {
                    if (i == AR_PREVIEW_POINT_CLOUD)
                        continue;

                    std::string key = "Preview" + std::to_string(i + 1);
                    std::string source = std::string(ini.GetValue(cameraSerialNumber.c_str(), key.c_str(), ""));
                    std::string remoteHeadIndex = source.substr(source.find_last_of(':') + 1, source.length());
                    if (!source.empty()) {

                        dev.win[i].selectedSource = source.substr(0, source.find_last_of(':'));
                        dev.win[i].selectedRemoteHeadIndex = std::stoi(remoteHeadIndex);

                        Log::Logger::getInstance()->info(
                                ".ini file: found source '{}' for preview {} at head {}, Adding to requested source",
                                source.substr(0, source.find_last_of(':')),
                                i + 1, ch);

                        chInfo.requestedStreams.emplace_back(dev.win[i].selectedSource);

                    }
                }
            }
        }

        dev.channelInfo.at(ch) = chInfo;
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

        WinRegEditor regEditor(dev.interfaceName, dev.interfaceDescription, dev.interfaceIndex);
        if (regEditor.ready) {
            regEditor.readAndBackupRegisty();
            regEditor.setTCPIPValues(hostAddress, "255.255.255.0");
            regEditor.setJumboPacket("9014");
            regEditor.restartNetAdapters();
            std::this_thread::sleep_for(std::chrono::milliseconds(8000));
            // TODO
            // 8 Seconds to wait for adapter to restart. This will vary from machine to machine and should be re-done
            // If possible then wait for a windows event that triggers when the adapter is ready
        }


#else
        int ioctl_result = -1;
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

        /*** Call ioctl to get network device configuration ***/
        /*
        std::string ipAddressBackup = "";
        std::string subnetMaskBackup = "";
        uint32_t mtuSizeBackup = 1500;
        ioctl_result = ioctl(sd, SIOCGIFADDR, &ifr);  // Set IP address
        if (ioctl_result < 0) {
            fprintf(stderr, "ioctl SIOCGIFADDR: %s", strerror(errno));
            if (errno == EADDRNOTAVAIL) {
                Log::Logger::getInstance()->error("No address is set on interface {}", dev.interfaceName);
            }
        } else {
            ipAddressBackup = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);
        }
        ioctl_result = ioctl(sd, SIOCGIFNETMASK, &ifr);   // Set subnet mask
        if (ioctl_result < 0) {
            fprintf(stderr, "ioctl SIOCGIFNETMASK: %s", strerror(errno));
            if (errno == EADDRNOTAVAIL) {
                Log::Logger::getInstance()->error("No NetMask is set on interface {}", dev.interfaceName);
            }
        } else {
            subnetMaskBackup = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);
        }
        if (ioctl(sd, SIOCGIFMTU, (caddr_t) &ifr) < 0) {
            Log::Logger::getInstance()->error("Failed to get mtu size {} on adapter {}",
                                              dev.interfaceName.c_str());
        } else {
            Log::Logger::getInstance()->error("got MTU {} on adapter {}", ifr.ifr_mtu,
                                              dev.interfaceName.c_str());
            mtuSizeBackup = ifr.ifr_mtu;
        }

        // Save backup to file

        // Write to ini file.
        CSimpleIniA ini;
        ini.SetUnicode();
        SI_Error rc = ini.LoadFile("NetConfigBackup.ini");
        if (rc < 0) {
            // File doesn't exist error, then create one
            if (rc == SI_FILE && errno == ENOENT) {
                std::ofstream output = std::ofstream("NetConfigBackup.ini");
                output.close();
                rc = ini.LoadFile("NetConfigBackup.ini");
            }
        }
        int ret;
        if (!ini.SectionExists(dev.interfaceName.c_str())) {
            ret = ini.SetValue(dev.interfaceName.c_str(), "IPAddress", ipAddressBackup.c_str());
            ret = ini.SetValue(dev.interfaceName.c_str(), "SubnetMask", subnetMaskBackup.c_str());
            ret = ini.SetValue(dev.interfaceName.c_str(), "MTU", std::to_string(mtuSizeBackup).c_str());
            rc = ini.SaveFile("NetConfigBackup.ini");
        }


        /*** Call ioctl to get configure network interface ***/

        /// note: prepare the two struct sockaddr_in
        inet_addr.sin_family = AF_INET;
        int inet_addr_config_result = inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

        subnet_mask.sin_family = AF_INET;
        int subnet_mask_config_result = inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));

        /// put addr in ifr structure
        memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
        ioctl_result = ioctl(sd, SIOCSIFADDR, &ifr);  // Set IP address
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
    }
    return true;
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


void CameraConnection::connectCRLCameraTask(void *context, MultiSense::Device *dev, bool isRemoteHead,
                                            bool shouldConfigNetwork) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    app->setNetworkAdapterParameters(*dev, shouldConfigNetwork);

    // 1. Connect to camera
    // 2. If successful: Disable any other available camera
    Log::Logger::getInstance()->info("Connect.");
    Log::Logger::getInstance()->info("Creating new physical camera.");
    app->camPtr = std::make_unique<CRLPhysicalCamera>();
    // If we successfully connect
    dev->channelConnections = app->camPtr->connect(dev->IP, isRemoteHead);

    if (!dev->channelConnections.empty()) {
        app->setStreamingModes(*dev); // TODO Race condition in altering the *dev piece of memory
        dev->cameraName = app->camPtr->getCameraInfo(0).devInfo.name;
        dev->serialName = app->camPtr->getCameraInfo(0).devInfo.serialNumber;
        app->lastActiveDevice = dev->name;
        dev->state = AR_STATE_ACTIVE;
    } else {
        dev->state = AR_STATE_UNAVAILABLE;
        app->lastActiveDevice = "-1";
    }


}

void CameraConnection::saveProfileAndDisconnect(MultiSense::Device *dev) {
    Log::Logger::getInstance()->info("Disconnecting profile {} using camera {}", dev->name.c_str(),
                                     dev->cameraName.c_str());
    // Save settings to file. Attempt to create a new file if it doesn't exist
    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile("crl.ini");
    if (rc < 0) {
        // File doesn't exist error, then create one
        if (rc == SI_FILE && errno == ENOENT) {
            std::ofstream output("crl.ini");
            output.close();
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
        // Preview Data per channel
        for (const auto &ch: dev->channelConnections) {
            std::string mode = "Mode" + std::to_string(ch);
            addIniEntry(&ini, CRLSerialNumber, mode, std::to_string((int) Utils::stringToCameraResolution(
                    dev->channelInfo[ch].modes[dev->channelInfo[ch].selectedModeIndex])
            ));
            addIniEntry(&ini, CRLSerialNumber, "Layout", std::to_string((int) dev->layout));
            auto &chInfo = dev->channelInfo[ch];
            for (int i = 0; i < AR_PREVIEW_TOTAL_MODES; ++i) {
                if (i == AR_PREVIEW_POINT_CLOUD)
                    continue;
                std::string source = dev->win[i].selectedSource;
                std::string remoteHead = std::to_string(dev->win[i].selectedRemoteHeadIndex);
                std::string key = "Preview" + std::to_string(i + 1);
                std::string value = (source + ":" + remoteHead);
                addIniEntry(&ini, CRLSerialNumber, key, value);
            }
        }
    }
    // delete entry if we gave the disconnect and reset flag Otherwise just normal disconnect
    // save the data back to the file
    lastActiveDevice = "-1";
    if (dev->state == AR_STATE_DISCONNECT_AND_FORGET) {
        int done = ini.Delete(CRLSerialNumber.c_str(), nullptr);
        dev->state = AR_STATE_REMOVE_FROM_LIST;
        Log::Logger::getInstance()->info("Deleted saved profile for serial: {}", CRLSerialNumber);

    } else {
        dev->state = AR_STATE_DISCONNECTED;
    }
    rc = ini.SaveFile("crl.ini");
    if (rc < 0) {
        Log::Logger::getInstance()->info("Failed to save crl.ini file. Err: {}", rc);
    }
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
                                                   bool hdr, uint32_t index,
                                                   MultiSense::Device *dev) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setGamma(gamma);
    app->camPtr->setGain(gain);
    app->camPtr->setFps(fps, index);
    app->camPtr->setPostFilterStrength(spfs);
    app->camPtr->setHDR(hdr);
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