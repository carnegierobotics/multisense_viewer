//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECT_H
#define AUTOCONNECT_AUTOCONNECT_H

#define MAX_CONNECTION_ATTEMPTS 3
#define TIMEOUT_INTERVAL_SECONDS 10

#include <include/MultiSense/MultiSenseTypes.hh>
#include "include/MultiSense/MultiSenseChannel.hh"

#include <string>
#include <vector>
#include <thread>

class AutoConnect {
public:

    enum FoundCameraOnIp {
        FOUND_CAMERA = 0,
        NO_CAMERA_RETRY = 1,
        NO_CAMERA = 2
    };

    struct Result {
        std::string cameraIpv4Address;
        std::string networkAdapter;
        std::string networkAdapterLongName;
        uint32_t index;
    }result;

    struct AdapterSupportResult {
        std::string name; // Name of network adapter tested
        std::string description = "No description available";
        std::string lName;
        uint32_t index;
        bool supports; // 0: for bad, 1: for good

        AdapterSupportResult(const char *name, uint8_t supports) {
            this->name = name;
            this->supports = supports;
        }

        bool searched = false;
    };

    bool success = false;
    bool loopAdapters = true;
    bool listenOnAdapter = true;
    time_t startTime{};
    crl::multisense::Channel* cameraInterface{};

    std::vector<AdapterSupportResult> ignoreAdapters;

    virtual std::vector<AutoConnect::AdapterSupportResult> findEthernetAdapters(bool b, bool skipIgnored) = 0;
    virtual void start(std::vector<AdapterSupportResult> vector) = 0;
    virtual void onFoundAdapters(std::vector<AdapterSupportResult> vector, bool logEvent) = 0;
    virtual FoundCameraOnIp onFoundIp(std::string string, AdapterSupportResult adapter) = 0;
    virtual void onFoundCamera(AdapterSupportResult supportResult) = 0;
    virtual void stop() = 0;

    bool shouldProgramRun = false;

    virtual bool shouldProgramClose() = 0;

    virtual void setShouldProgramClose(bool exit) = 0;

private:
    struct CameraInfo {
        crl::multisense::system::DeviceInfo devInfo;
        crl::multisense::image::Config imgConf;
        crl::multisense::system::NetworkConfig netConfig;
        crl::multisense::system::VersionInfo versionInfo;
        crl::multisense::image::Calibration camCal{};
        std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes;
        crl::multisense::DataSource supportedSources{0};
        std::vector<uint8_t *> rawImages;
        int sensorMTU = 0;

    } cameraInfo;
protected:
    std::thread *t = nullptr;
    int connectAttemptCounter = 0;

};


#endif //AUTOCONNECT_AUTOCONNECT_H
