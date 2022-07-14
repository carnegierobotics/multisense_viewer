//
// Created by magnus on 7/14/22.
//

#ifndef AUTOCONNECT_AUTOCONNECT_H
#define AUTOCONNECT_AUTOCONNECT_H


#include <MultiSense/MultiSenseTypes.hh>

class AutoConnect {

public:
    struct AdapterSupportResult {
        std::string name; // Name of network adapter tested
        bool supports; // 0: for bad, 1: for good

        AdapterSupportResult(const char *name, uint8_t supports) {
            this->name = name;
            this->supports = supports;
        }
    };

    virtual std::vector<AdapterSupportResult> findEthernetAdapters() = 0;
    virtual void start(std::vector<AdapterSupportResult> vector) = 0;
    virtual void onFoundAdapters(std::vector<AdapterSupportResult> vector) = 0;
    virtual bool onFoundIp(std::string string, AdapterSupportResult adapter) = 0;
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

};


#endif //AUTOCONNECT_AUTOCONNECT_H
