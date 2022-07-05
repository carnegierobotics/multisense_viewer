//
// Created by magnus on 5/25/22.
//

#ifndef MULTISENSE_VIEWER_CRLBASEINTERFACE_H
#define MULTISENSE_VIEWER_CRLBASEINTERFACE_H
#include <MultiSense/MultiSenseChannel.hh>
#include <unordered_map>

class CRLBaseInterface {

public:

    virtual ~CRLBaseInterface() {
        printf("BaseInterface destructor\n");
    };

    struct CameraInfo {
        crl::multisense::system::DeviceInfo devInfo;
        crl::multisense::image::Config imgConf;
        crl::multisense::system::NetworkConfig netConfig;
        crl::multisense::system::VersionInfo versionInfo;
        crl::multisense::image::Calibration camCal{};
        std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes;
        crl::multisense::DataSource supportedSources{0};
        std::vector<uint8_t*> rawImages;
        int sensorMTU = 0;
    }cameraInfo;

    virtual bool connect(const std::string& ip) = 0;
    virtual void updateCameraInfo() = 0;
    virtual void start(std::string string, std::string dataSourceStr) = 0;
    virtual void stop(std::string dataSourceStr) = 0;
    virtual CameraInfo getCameraInfo() {
        return cameraInfo;
    }

    std::unordered_map<crl::multisense::DataSource, crl::multisense::image::Header> getImage() {
        return imagePointers;
    }

    std::unordered_map<crl::multisense::DataSource,crl::multisense::image::Header> imagePointers;
    bool online = false;
    bool modeChange = false;
    bool play = false;
    std::vector<crl::multisense::DataSource> enabledSources;


};


#endif //MULTISENSE_VIEWER_CRLBASEINTERFACE_H
