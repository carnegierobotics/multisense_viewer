//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECT_H
#define AUTOCONNECT_AUTOCONNECT_H

#define MAX_CONNECTION_ATTEMPTS 2
#define TIMEOUT_INTERVAL_SECONDS 5

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

        Result() = default;

        Result(const char *name, uint8_t supports) {
            this->networkAdapter = name;
            this->supports = supports;
        }
        bool supports{}; // 0: for bad, 1: for good
        bool searched = false;
        std::string cameraIpv4Address;
        std::string description = "No description available";
        std::string networkAdapter;
        std::string networkAdapterLongName;
        uint32_t index{};

    }result;


    bool success = false;
    bool loopAdapters = true;
    bool listenOnAdapter = true;
    time_t startTime{};
    crl::multisense::Channel* cameraInterface{};

    std::vector<Result> ignoreAdapters;

    virtual std::vector<AutoConnect::Result> findEthernetAdapters(bool b, bool skipIgnored) = 0;
    virtual void start(std::vector<Result> vector) = 0;
    virtual void onFoundAdapters(std::vector<Result> vector, bool logEvent) = 0;
    virtual AutoConnect::FoundCameraOnIp onFoundIp(std::string string, Result adapter, int camera_fd) = 0;
    virtual void onFoundCamera() = 0;
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
