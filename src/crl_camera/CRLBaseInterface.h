//
// Created by magnus on 5/25/22.
//

#ifndef MULTISENSE_CRLBASEINTERFACE_H
#define MULTISENSE_CRLBASEINTERFACE_H

#include <MultiSense/src/core/Definitions.h>

#include <include/MultiSense/MultiSenseChannel.hh>
#include <unordered_map>
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

    virtual bool start(CRLCameraResolution resolution, std::string string) = 0;
    virtual void start(std::string src, StreamIndex parent){};

    virtual bool stop(std::string dataSourceStr) = 0;
    virtual void stop(StreamIndex parent)  {};

    virtual void preparePointCloud(uint32_t width, uint32_t height) = 0;

    virtual CameraInfo getCameraInfo() = 0;


    virtual bool getCameraStream(ArEngine::YUVTexture *tex) { return false;};
    virtual bool getCameraStream(std::string stringSrc, ArEngine::TextureData *tex) { return  false;};
    virtual bool getCameraStream(ArEngine::MP4Frame* frame, StreamIndex parent) { return false; };


    virtual void setExposure(uint32_t e){}
    virtual void setExposureParams(ExposureParams p){}

    virtual void setWhiteBalance(WhiteBalanceParams param) {}
    virtual void setPostFilterStrength(float filterStrength) {}
    virtual void setGamma(float gamma) {}
    virtual void setFps(float fps) {}
    virtual void setGain(float gain) {}
    virtual void setResolution(CRLCameraResolution res) {}

};


#endif //MULTISENSE_CRLBASEINTERFACE_H
