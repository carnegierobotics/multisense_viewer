//
// Created by magnus on 5/25/22.
//

#ifndef MULTISENSE_VIEWER_CRLBASEINTERFACE_H
#define MULTISENSE_VIEWER_CRLBASEINTERFACE_H

#include <MultiSense/MultiSenseChannel.hh>
#include <unordered_map>
#include <MultiSense/src/core/Definitions.h>
#include "glm/glm.hpp"

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
        std::vector<uint8_t *> rawImages;
        int sensorMTU = 0;
        glm::mat4 kInverseMatrix;
    };


    virtual bool connect(const std::string &ip) = 0;

    virtual void updateCameraInfo() = 0;

    virtual void start(std::string string, std::string dataSourceStr) = 0;
    virtual void start(std::string src, StreamIndex parent){};

    virtual void stop(std::string dataSourceStr) = 0;
    virtual void stop(StreamIndex parent)  {};

    virtual void preparePointCloud(uint32_t width, uint32_t height) = 0;

    virtual CameraInfo getCameraInfo() = 0;


    virtual void getCameraStream(std::string stringSrc, crl::multisense::image::Header *src,
                                 crl::multisense::image::Header **src2 = nullptr) {};
    virtual void getCameraStream(crl::multisense::image::Header *stream) {};
    virtual bool getCameraStream(ArEngine::MP4Frame* frame, StreamIndex parent) { return false; };


    virtual void setExposure(uint32_t e){}
    virtual void setExposureParams(ExposureParams p){}

    virtual void setWhiteBalance(WhiteBalanceParams param) {}
    virtual void setPostFilterStrength(float filterStrength) {}
    virtual void setGamma(float gamma) {}
    virtual void setFps(float fps) {}
    virtual void setGain(float gain) {}

};


#endif //MULTISENSE_VIEWER_CRLBASEINTERFACE_H
