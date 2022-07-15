//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECT_H
#define AUTOCONNECT_AUTOCONNECT_H

#define MAX_CONNECTION_ATTEMPTS 4
#define TIMEOUT_INTERVAL_SECONDS 6

#include <include/MultiSense/MultiSenseTypes.hh>
#include "MultiSense/MultiSenseChannel.hh"

#include <string>
#include <vector>
#include <thread>

class AutoConnect {
public:

    enum FoundCameraOnIp {
        FOUND_CAMERA = 1,
        NO_CAMERA_RETRY = 0,
        NO_CAMERA = -1
    };

    struct Result {
        std::string cameraIpv4Address;
        std::string networkAdapter;
        std::string networkAdapterLongName;
    }result;

    struct AdapterSupportResult {
        std::string name; // Name of network adapter tested
        std::string description = "No description available";
        std::string lName;
        bool supports; // 0: for bad, 1: for good

        AdapterSupportResult(const char *name, uint8_t supports) {
            this->name = name;
            this->supports = supports;
        }
    };

    bool success = false;
    bool loopAdapters = true;
    bool listenOnAdapter = true;
    time_t startTime;
    crl::multisense::Channel* cameraInterface;

    virtual std::vector<AdapterSupportResult> findEthernetAdapters() = 0;
    virtual void start(std::vector<AdapterSupportResult> vector) = 0;
    virtual void onFoundAdapters(std::vector<AdapterSupportResult> vector) = 0;
    virtual FoundCameraOnIp onFoundIp(std::string string, AdapterSupportResult adapter) = 0;
    virtual void onFoundCamera() = 0;
    virtual void stop() = 0;

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
    std::thread *t{};
    int connectAttemptCounter = 0;

};


#endif //AUTOCONNECT_AUTOCONNECT_H
