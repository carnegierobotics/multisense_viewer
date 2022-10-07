//
// Created by magnus on 3/1/22.
//
#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include "CRLPhysicalCamera.h"

#include <vulkan/vulkan_core.h>
#include "MultiSense/src/Tools/Utils.h"


std::vector<uint32_t> CRLPhysicalCamera::connect(const std::string &ip, bool isRemoteHead) {
    std::vector<uint32_t> indices;
    // If RemoteHead then attempt to connect 4 LibMultiSense channels
    // else create only one and place it at 0th index.
    for (uint32_t i = 0; i < (isRemoteHead ? (crl::multisense::Remote_Head_3 + 1) : 1); ++i) {
        channelMap[i] = isRemoteHead ? std::make_unique<ChannelWrapper>(ip, i) : std::make_unique<ChannelWrapper>(ip);
        if (channelMap[i] != nullptr) {
            updateCameraInfo(i);
            setMtu(7200, i);
            // Start some timers
            callbackTime = std::chrono::steady_clock::now();
            startTime = std::chrono::steady_clock::now();
            startTimeImu = std::chrono::steady_clock::now();
            addCallbacks(i);
            indices.emplace_back(i);
        }
    }
    return indices;
}

bool CRLPhysicalCamera::start(const std::string &dataSourceStr, uint32_t channelID) {
    crl::multisense::DataSource source = Utils::stringToDataSource(dataSourceStr);
    if (source == false)
        return false;
    // Start stream
    int32_t status = channelMap[channelID]->ptr()->startStreams(source);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Enabled stream: {} at remote head {}",
                                         Utils::dataSourceToString(source).c_str(), channelID);
        return true;
    } else
        Log::Logger::getInstance()->info("Failed to flashing stream: {}  status code {}",
                                         Utils::dataSourceToString(source).c_str(), status);
    return false;
}

bool CRLPhysicalCamera::stop(const std::string &dataSourceStr, uint32_t channelID) {
    if (channelMap[channelID] == nullptr)
        return false;
    crl::multisense::DataSource src = Utils::stringToDataSource(dataSourceStr);
    bool status = channelMap[channelID]->ptr()->stopStreams(src);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Stopped camera stream {}", dataSourceStr.c_str());
        return true;
    } else {
        Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
        return false;
    }
}

void CRLPhysicalCamera::remoteHeadOneCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto p = reinterpret_cast<CRLPhysicalCamera *>(userDataP);
    p->callbackTime = std::chrono::steady_clock::now();
    p->startTime = std::chrono::steady_clock::now();
    p->channelMap[0]->dataBase->updateImageBuffer(std::make_shared<ImageBufferWrapper>(p->channelMap[0]->ptr(), header));
}

void CRLPhysicalCamera::addCallbacks(uint32_t idx) {
    for (const auto &e: infoMap[idx].supportedDeviceModes)
        infoMap[idx].supportedSources |= e.supportedDataSources;
    // reserve double_buffers for each stream
    uint_fast8_t num_sources = 0;
    crl::multisense::DataSource d = infoMap[idx].supportedSources;
    while (d) {
        num_sources += (d & 1);
        d >>= 1;
    }
    if (channelMap[idx]->ptr()->addIsolatedCallback(remoteHeadOneCallback, infoMap[idx].supportedSources, this) !=
        crl::multisense::Status_Ok)
        std::cerr << "Adding callback failed!\n";
}

CRLPhysicalCamera::CameraInfo CRLPhysicalCamera::getCameraInfo(uint32_t idx) {
    return infoMap[idx];
}

bool CRLPhysicalCamera::getCameraStream(std::string stringSrc, VkRender::TextureData *tex, uint32_t idx) {
    // TODO Fix with proper conditions for checking if a frame is good or not

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span =
            std::chrono::duration_cast<std::chrono::duration<float>>(time - startTime);
    if (time_span.count() > 1) {
        std::chrono::duration<float> callbackTimeSpan =
                std::chrono::duration_cast<std::chrono::duration<float>>(time - callbackTime);
        Log::Logger::getInstance()->info(
                "Requesting {} from head {}. It is {}s since last call",
                stringSrc, idx, callbackTimeSpan.count());
        startTime = std::chrono::steady_clock::now();
    }
    auto src = Utils::stringToDataSource(stringSrc);

    crl::multisense::DataSource colorSource;
    crl::multisense::DataSource lumaSource;
    std::shared_ptr<ImageBufferWrapper> header;
    std::shared_ptr<ImageBufferWrapper> headerTwo;
    if (stringSrc == "Color Aux") {
        colorSource = crl::multisense::Source_Chroma_Aux;
        lumaSource = crl::multisense::Source_Luma_Aux;
        header = channelMap[0]->dataBase->getImageBuffer(idx, lumaSource);
        headerTwo = channelMap[0]->dataBase->getImageBuffer(idx, colorSource);
        if (headerTwo == nullptr)
            return false;

    } else if (stringSrc == "Color Rectified Aux") {
        colorSource = crl::multisense::Source_Chroma_Rectified_Aux;
        lumaSource = crl::multisense::Source_Luma_Rectified_Aux;
        header = channelMap[0]->dataBase->getImageBuffer(idx, lumaSource);
        headerTwo = channelMap[0]->dataBase->getImageBuffer(idx, colorSource);
        if (headerTwo == nullptr)
            return false;
    } else {
        header = channelMap[0]->dataBase->getImageBuffer(idx, src);
    }

    if (header == nullptr)
        return false;

    switch (tex->type) {
        case AR_COLOR_IMAGE_YUV420:
            std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
            std::memcpy(tex->data2, headerTwo->data().imageDataP, headerTwo->data().imageLength);
            return true;

        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_DISPARITY_IMAGE:
            if (header->data().bitsPerPixel != 16) {
                std::cerr << "Unsupported disparity pixel depth" << std::endl;
                break;
            }
        case AR_GRAYSCALE_IMAGE:
        case AR_POINT_CLOUD:
            if (header->data().source != src)
                return false;
            std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
            return true;
        case AR_CAMERA_IMAGE_NONE:
            break;
    }
    return false;
}


void CRLPhysicalCamera::preparePointCloud(uint32_t width, uint32_t height) {
    const double xScale = 1.0 / ((static_cast<double>(infoMap[0].devInfo.imagerWidth) /
                                  static_cast<double>(width)));
    // From LibMultisenseUtility
    crl::multisense::image::Config c = infoMap[0].imgConf;
    const double fx = c.fx();
    const double fy = c.fy();
    const double cx = c.cx();
    const double cy = c.cy();
    const double tx = c.tx();
    const double cxRight = infoMap[0].calibration.right.P[0][2] * xScale;

    glm::mat4 kInverseMatrix(
            glm::vec4(fy * tx, 0, 0, -fy * cx * tx),
            glm::vec4(0, fx * tx, 0, -fx * cy * tx),
            glm::vec4(0, 0, 0, fx * fy * tx),
            glm::vec4(0, 0, -fy, fy * (cx - cxRight)));

    infoMap[0].kInverseMatrix = glm::transpose(kInverseMatrix);
}


void CRLPhysicalCamera::updateCameraInfo(uint32_t idx) {
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getImageConfig(infoMap[idx].imgConf)) {
        Log::Logger::getInstance()->info("Failed to update Light config");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getNetworkConfig(infoMap[idx].netConfig)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "netConfig");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getVersionInfo(infoMap[idx].versionInfo)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "versionInfo");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getDeviceInfo(infoMap[idx].devInfo)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "devInfo");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getDeviceModes(infoMap[idx].supportedDeviceModes)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedDeviceModes");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getImageCalibration(infoMap[idx].camCal)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "camCal");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getEnabledStreams(infoMap[idx].supportedSources)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedSources");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getMtu(infoMap[idx].sensorMTU)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "sensorMTU");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getLightingConfig(infoMap[idx].lightConf)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "lightConf");
        return;
    }

    if (crl::multisense::Status_Ok != channelMap[idx]->ptr()->getImageCalibration(infoMap[idx].calibration)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "calibration");
        return;
    }

}


void CRLPhysicalCamera::setGamma(float gamma) {
    crl::multisense::Status status = channelMap[0]->ptr()->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query gamma configuration");
    }
    infoMap[0].imgConf.setGamma(gamma);
    status = channelMap[0]->ptr()->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set gamma configuration");
    }

    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setFps(float fps, uint32_t channelID) {
    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    infoMap[channelID].imgConf.setFps(fps);

    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setGain(float gain) {
    crl::multisense::Status status = channelMap[0]->ptr()->getImageConfig(infoMap[0].imgConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }

    infoMap[0].imgConf.setGain(gain);

    status = channelMap[0]->ptr()->setImageConfig(infoMap[0].imgConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);
}


void CRLPhysicalCamera::setResolution(CRLCameraResolution resolution, uint32_t channelID) {

    if (resolution == currentResolutionMap[channelID] || resolution == CRL_RESOLUTION_NONE)
        return;
    uint32_t width = 0, height = 0, depth = 0;
    Utils::cameraResolutionToValue(resolution, &width, &height, &depth);
    if (width == 0 || height == 0 || depth == 0) {
        Log::Logger::getInstance()->error("Resolution mode not supported");
        return;
    }
    int ret = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (ret != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->error("failed to get image config");
    }
    if (Utils::valueToCameraResolution(infoMap[channelID].imgConf.width(), infoMap[channelID].imgConf.height(),
                                       infoMap[channelID].imgConf.disparities()) ==
        resolution) {
        currentResolutionMap[channelID] = resolution;
        Log::Logger::getInstance()->info("Resolution already set to {}x{}x{}", width, height, depth);
        return;
    }

    infoMap[channelID].imgConf.setResolution(width, height);
    infoMap[channelID].imgConf.setDisparities(depth);
    ret = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{} on channel {}", width, height, depth, channelID);
        currentResolutionMap[channelID] = resolution;
    } else {
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height, depth, ret);
        return;
    }

    this->updateCameraInfo(channelID);
}

void CRLPhysicalCamera::setExposureParams(ExposureParams p) {

    crl::multisense::Status status = channelMap[0]->ptr()->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    if (p.autoExposure) {
        infoMap[0].imgConf.setAutoExposure(p.autoExposure);
        infoMap[0].imgConf.setAutoExposureMax(p.autoExposureMax);
        infoMap[0].imgConf.setAutoExposureDecay(p.autoExposureDecay);
        infoMap[0].imgConf.setAutoExposureTargetIntensity(p.autoExposureTargetIntensity);
        infoMap[0].imgConf.setAutoExposureThresh(p.autoExposureThresh);
    } else {
        infoMap[0].imgConf.setAutoExposure(p.autoExposure);
        infoMap[0].imgConf.setExposure(p.exposure);
    }

    infoMap[0].imgConf.setExposureSource(p.exposureSource);
    status = channelMap[0]->ptr()->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setPostFilterStrength(float filter) {
    crl::multisense::Status status = channelMap[0]->ptr()->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    infoMap[0].imgConf.setStereoPostFilterStrength(filter);
    status = channelMap[0]->ptr()->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }
    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setHDR(bool hdr) {
    crl::multisense::Status status = channelMap[0]->ptr()->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query hdr image configuration");
    }

    infoMap[0].imgConf.setHdr(hdr);
    status = channelMap[0]->ptr()->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set hdr configuration");
    }
    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setWhiteBalance(WhiteBalanceParams param) {
    crl::multisense::Status status = channelMap[0]->ptr()->getImageConfig(infoMap[0].imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    if (param.autoWhiteBalance) {
        infoMap[0].imgConf.setAutoWhiteBalance(param.autoWhiteBalance);
        infoMap[0].imgConf.setAutoWhiteBalanceThresh(param.autoWhiteBalanceThresh);
        infoMap[0].imgConf.setAutoWhiteBalanceDecay(param.autoWhiteBalanceDecay);

    } else {
        infoMap[0].imgConf.setAutoWhiteBalance(param.autoWhiteBalance);
        infoMap[0].imgConf.setWhiteBalance(param.whiteBalanceRed, param.whiteBalanceBlue);
    }
    status = channelMap[0]->ptr()->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);

}

void CRLPhysicalCamera::setLighting(LightingParams param) {
    crl::multisense::Status status = channelMap[0]->ptr()->getLightingConfig(infoMap[0].lightConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }

    if (param.selection != -1) {
        infoMap[0].lightConf.setDutyCycle(param.selection, param.dutyCycle);
    } else
        infoMap[0].lightConf.setDutyCycle(param.dutyCycle);

    infoMap[0].lightConf.setNumberOfPulses(param.numLightPulses);
    infoMap[0].lightConf.setLedStartupTime(param.startupTime);
    infoMap[0].lightConf.setFlash(param.flashing);
    status = channelMap[0]->ptr()->setLightingConfig(infoMap[0].lightConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }


}

void CRLPhysicalCamera::setMtu(uint32_t mtu, uint32_t channelID) {
    int status = channelMap[channelID]->ptr()->setMtu(mtu);
    if (status != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Failed to set MTU {}", mtu);
    } else {
        Log::Logger::getInstance()->info("Set MTU to {}", mtu);
    }
}
