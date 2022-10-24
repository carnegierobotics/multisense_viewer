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

#include <MultiSense/Src/Tools/Utils.h>
#include "MultiSense/external/simpleini/SimpleIni.h"

#define MAX_TASK_STACK_SIZE 3
#define MAX_NUM_REMOTEHEADS 4

void CameraConnection::updateActiveDevice(VkRender::Device *dev) {
    auto *p = &dev->parameters;
    if (dev->parameters.updateGuiParams) {
        const auto &conf = camPtr->getCameraInfo(dev->configRemoteHead).imgConf;
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

        const auto &lightConf = camPtr->getCameraInfo(dev->configRemoteHead).lightConf;
        p->light.numLightPulses = lightConf.getNumberOfPulses();
        p->light.dutyCycle = lightConf.getDutyCycle(0);
        p->light.flashing = lightConf.getFlash();
        p->light.startupTime = lightConf.getStartupTime();

        p->stereoPostFilterStrength = conf.stereoPostFilterStrength();

        dev->parameters.updateGuiParams = false;
    }

    if (dev->parameters.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
        pool->Push(CameraConnection::setAdditionalParametersTask, this, p->fps, p->gain, p->gamma,
                   p->stereoPostFilterStrength, p->hdrEnabled, dev, dev->configRemoteHead);

    if (dev->parameters.ep.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
        pool->Push(CameraConnection::setExposureTask, this, &p->ep, dev, dev->configRemoteHead);

    if (dev->parameters.wb.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
        pool->Push(CameraConnection::setWhiteBalanceTask, this, &p->wb, dev, dev->configRemoteHead);

    if (dev->parameters.light.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
        pool->Push(CameraConnection::setLightingTask, this, &p->light, dev, dev->configRemoteHead);
    // Set the correct resolution. Will only update if changed.

    for (auto &ch: dev->channelInfo) {
        if (ch.state == AR_STATE_ACTIVE && ch.updateResolutionMode && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            pool->Push(CameraConnection::setResolutionTask, this, ch.selectedMode, ch.index);
            ch.updateResolutionMode = false;
        }

    }

    // Start requested streams
    for (auto &ch: dev->channelInfo) {
        if (ch.state != AR_STATE_ACTIVE && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
            continue;
        for (const auto &requested: ch.requestedStreams) {
            if (!Utils::isInVector(ch.enabledStreams, requested) && requested != "Source") {
                pool->Push(CameraConnection::startStreamTask, this, requested, ch.index);
                ch.enabledStreams.emplace_back(requested);
            }
        }
    }

    // Stop if the stream is no longer requested
    for (auto &ch: dev->channelInfo) {
        if (ch.state != AR_STATE_ACTIVE && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
            continue;
        for (const auto &enabled: ch.enabledStreams) {
            if (!Utils::isInVector(ch.requestedStreams, enabled) && !enabled.empty()) {
                pool->Push(CameraConnection::stopStreamTask, this, enabled, ch.index);
                Utils::removeFromVector(&ch.enabledStreams, enabled);
            }
        }
    }

}

void
CameraConnection::onUIUpdate(std::vector<VkRender::Device> &devices, bool shouldConfigNetwork, bool isRemoteHead) {
    // If no device is connected then return
    if (devices.empty())
        return;
    // Check for actions on each element
    for (auto &dev: devices) {

        // If we have requested to reset a connection or want to remove it from saved profiles
        if (dev.state == AR_STATE_RESET || dev.state == AR_STATE_DISCONNECT_AND_FORGET) {
            // Only call this task if it is not already running
            saveProfileAndDisconnect(&dev);
            if (dev.state == AR_STATE_RESET) {
                for (auto ch: dev.channelConnections)
                    camPtr->stop("All", ch); // This operation may be blocking because we dont want to stop our threadpool before we know camPtr->stop is finished

            }
            pool->Stop();
            return;
        }
        // Connect if we click a device in the sidebar or if it is just added by add device btn
        if ((dev.clicked && dev.state != AR_STATE_ACTIVE) || dev.state == AR_STATE_JUST_ADDED) {
            // reset other active device if present. So loop over all devices again.
            bool resetOtherDevice = false;
            VkRender::Device *otherDev;
            for (auto &d: devices) {
                if (d.state == AR_STATE_ACTIVE && d.name != dev.name) {
                    d.state = AR_STATE_RESET;
                    Log::Logger::getInstance()->info("Call to reset state requested for profile {}", d.name);
                    resetOtherDevice = true;
                    otherDev = &d;
                }
            }
            if (resetOtherDevice) {
                for (auto ch: dev.channelConnections)
                    camPtr->stop("All",
                                 ch); // Blocking operation because we don't want to make a new connection before we have stopped previous connection

                saveProfileAndDisconnect(otherDev);
            }
            dev.state = AR_STATE_CONNECTING;
            // Re-create thread pool for a new connection in case we have old tasks from another connection in queue
            pool = std::make_unique<VkRender::ThreadPool>(1);
            // Perform connection by pushing a connect task.
            pool->Push(CameraConnection::connectCRLCameraTask, this, &dev, isRemoteHead, shouldConfigNetwork);
            break;
        }

        // Rest of this function is only needed for active devices
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

void CameraConnection::getProfileFromIni(VkRender::Device &dev) {
    dev.channelInfo.resize(MAX_NUM_REMOTEHEADS); // max number of remote heads
    dev.win.clear();

    for (auto ch: dev.channelConnections) {
        VkRender::ChannelInfo chInfo;
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


bool CameraConnection::setNetworkAdapterParameters(VkRender::Device &dev, bool shouldConfigNetwork) {
    std::string hostAddress = dev.IP;
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
        if (regEditor.ready && !dev.systemNetworkChanged) {
            regEditor.readAndBackupRegisty();
            //regEditor.setTCPIPValues(hostAddress, "255.255.255.0");
            regEditor.setJumboPacket("9014");
            regEditor.restartNetAdapters();
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
            if (regEditor.setStaticIp(dev.interfaceIndex, hostAddress, "255.255.255.0")) {
                dev.systemNetworkChanged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
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

        /*** Call ioctl to get and backup current network device configuration ***/
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

        */
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
#ifndef WIN32
    if (sd != -1)
        close(sd);
#endif // !WIN32
}


void CameraConnection::connectCRLCameraTask(void *context, VkRender::Device *dev, bool isRemoteHead,
                                            bool shouldConfigNetwork) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    app->setNetworkAdapterParameters(*dev, shouldConfigNetwork);
    // Connect to camera
    Log::Logger::getInstance()->info("Creating connection to camera.");
    // Stop all tasks from previous connection.
    app->camPtr = std::make_unique<CRLPhysicalCamera>();
    // If we successfully connect
    dev->channelConnections = app->camPtr->connect(dev->IP, isRemoteHead);
    if (!dev->channelConnections.empty()) {
        app->getProfileFromIni(*dev);
        // Set the resolution read from config file
        dev->cameraName = app->camPtr->getCameraInfo(dev->channelConnections.front()).devInfo.name;
        dev->serialName = app->camPtr->getCameraInfo(dev->channelConnections.front()).devInfo.serialNumber;
        dev->state = AR_STATE_ACTIVE;
    } else {
        dev->state = AR_STATE_UNAVAILABLE;
    }


}

void CameraConnection::saveProfileAndDisconnect(VkRender::Device *dev) {
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
    if (dev->state == AR_STATE_DISCONNECT_AND_FORGET) {
        ini.Delete(CRLSerialNumber.c_str(), nullptr);
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

void CameraConnection::setExposureTask(void *context, ExposureParams *arg1, VkRender::Device *dev,
                                       crl::multisense::RemoteHeadChannel remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);

    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setExposureParams(*arg1, remoteHeadIndex);
    app->updateFromCameraParameters(dev, remoteHeadIndex);
}

void CameraConnection::setWhiteBalanceTask(void *context, WhiteBalanceParams *arg1, VkRender::Device *dev,
                                           crl::multisense::RemoteHeadChannel remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setWhiteBalance(*arg1, remoteHeadIndex);
    app->updateFromCameraParameters(dev, remoteHeadIndex);
}

void CameraConnection::setLightingTask(void *context, LightingParams *arg1, VkRender::Device *dev,
                                       crl::multisense::RemoteHeadChannel remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setLighting(*arg1, remoteHeadIndex);
    app->updateFromCameraParameters(dev, remoteHeadIndex);
}

void CameraConnection::setAdditionalParametersTask(void *context, float fps, float gain, float gamma, float spfs,
                                                   bool hdr, VkRender::Device *dev,
                                                   crl::multisense::RemoteHeadChannel index) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    if (!app->camPtr->setGamma(gamma, index))return;
    if (!app->camPtr->setGain(gain, index)) return;
    if (!app->camPtr->setFps(fps, index)) return;
    if (!app->camPtr->setPostFilterStrength(spfs, index)) return;
    if (!app->camPtr->setHDR(hdr, index)) return;
    app->updateFromCameraParameters(dev, index);
}

void
CameraConnection::setResolutionTask(void *context, CRLCameraResolution arg1, crl::multisense::RemoteHeadChannel idx) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    std::scoped_lock lock(app->writeParametersMtx);
    app->camPtr->setResolution(arg1, idx);
}

void CameraConnection::startStreamTask(void *context, std::string src,
                                       crl::multisense::RemoteHeadChannel remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    app->camPtr->start(src, remoteHeadIndex);
}

void CameraConnection::stopStreamTask(void *context, std::string src,
                                      crl::multisense::RemoteHeadChannel remoteHeadIndex) {
    auto *app = reinterpret_cast<CameraConnection *>(context);
    if (app->camPtr->stop(src, remoteHeadIndex));
    else
        Log::Logger::getInstance()->info("Failed to disable stream {}", src);
}

void CameraConnection::updateFromCameraParameters(VkRender::Device *dev,
                                                  crl::multisense::RemoteHeadChannel remoteHeadIndex) const {
    // Query the camera for new values and update the GUI. It is a way to see if the actual value was set.
    const auto &conf = camPtr->getCameraInfo(remoteHeadIndex).imgConf;
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
    dev->parameters.updateGuiParams = false;
}