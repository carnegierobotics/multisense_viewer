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
        channelMap[i] = isRemoteHead ? crl::multisense::Channel::Create(ip, i) : crl::multisense::Channel::Create(ip);
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
    int32_t status = channelMap[channelID]->startStreams(source);
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

    bool status = channelMap[channelID]->stopStreams(src);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Stopped camera stream {}", dataSourceStr.c_str());
        return true;
    } else {
        Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
        return false;
    }
}


void CRLPhysicalCamera::streamCallbackRemoteHead(const crl::multisense::image::Header &image, uint32_t idx) {

    auto &buf = buffersMap[idx][image.source];

    // TODO: make this a method of the BufferPair or something
    std::scoped_lock lock(buf.swap_lock);

    if (buf.inactiveCBBuf != nullptr)  // initial state
    {
        channelMap[idx]->releaseCallbackBuffer(buf.inactiveCBBuf);
    }


    if (image.imageDataP == nullptr) {
        Log::Logger::getInstance()->info("Image from camera was empty");
    } else
        imagePointersMap[idx][image.source] = image;

    buf.inactiveCBBuf = channelMap[idx]->reserveCallbackBuffer();
    buf.inactive = image;
}

void CRLPhysicalCamera::remoteHeadOneCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto cam = reinterpret_cast<CRLPhysicalCamera *>(userDataP);

    cam->callbackTime = std::chrono::steady_clock::now();
    cam->startTime = std::chrono::steady_clock::now();
    cam->streamCallbackRemoteHead(header, 0);
}

void CRLPhysicalCamera::remoteHeadTwoCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto cam = reinterpret_cast<CRLPhysicalCamera *>(userDataP);

    cam->callbackTime = std::chrono::steady_clock::now();
    cam->startTime = std::chrono::steady_clock::now();
    cam->streamCallbackRemoteHead(header, 1);
}

void CRLPhysicalCamera::remoteHeadThreeCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto cam = reinterpret_cast<CRLPhysicalCamera *>(userDataP);

    cam->callbackTime = std::chrono::steady_clock::now();
    cam->startTime = std::chrono::steady_clock::now();
    cam->streamCallbackRemoteHead(header, 2);
}

void CRLPhysicalCamera::remoteHeadFourCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto cam = reinterpret_cast<CRLPhysicalCamera *>(userDataP);

    cam->callbackTime = std::chrono::steady_clock::now();
    cam->startTime = std::chrono::steady_clock::now();
    cam->streamCallbackRemoteHead(header, 3);
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

    switch (idx) {
        case 0:
            if (channelMap[idx]->addIsolatedCallback(remoteHeadOneCallback, infoMap[idx].supportedSources, this) !=
                crl::multisense::Status_Ok)
                std::cerr << "Adding callback failed!\n";
            break;
        case 1:
            if (channelMap[idx]->addIsolatedCallback(remoteHeadTwoCallback, infoMap[idx].supportedSources, this) !=
                crl::multisense::Status_Ok)
                std::cerr << "Adding callback failed!\n";
            break;
        case 2:
            if (channelMap[idx]->addIsolatedCallback(remoteHeadThreeCallback, infoMap[idx].supportedSources, this) !=
                crl::multisense::Status_Ok)
                std::cerr << "Adding callback failed!\n";
            break;
        case 3:
            if (channelMap[idx]->addIsolatedCallback(remoteHeadFourCallback, infoMap[idx].supportedSources, this) !=
                crl::multisense::Status_Ok)
                std::cerr << "Adding callback failed!\n";
            break;
    }
}

CRLPhysicalCamera::CameraInfo CRLPhysicalCamera::getCameraInfo(uint32_t idx) {
    return infoMap[idx];
}

bool CRLPhysicalCamera::getImuRotation(VkRender::Rotation *rot) {

    auto lock = std::scoped_lock<std::mutex>(swap_lock);
    rot->roll = rotationData.roll;
    rot->pitch = rotationData.pitch;

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span =
            std::chrono::duration_cast<std::chrono::duration<float>>(time - startTimeImu);
    if (time_span.count() > 1) {
        Log::Logger::getInstance()->info(
                "Requesting imu data from camera. But it haven't been updated in a while");
        startTimeImu = std::chrono::steady_clock::now();
        rotationData.roll = 0;
        rotationData.pitch = 0;
        rotationData.yaw = 0;
        return false;
    }

    return true;
}


/*
bool CRLPhysicalCamera::getCameraStream(VkRender::YUVTexture *tex) {
    assert(tex != nullptr);
    tex->format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;

    auto &chroma = imagePointers[crl::multisense::Source_Chroma_Rectified_Aux];
    if (chroma.imageDataP != nullptr && chroma.source == crl::multisense::Source_Chroma_Rectified_Aux) {
        tex->data[0] = malloc(chroma.imageLength); // TODO leverage STL... Rethink malloc
        memcpy(tex->data[0], chroma.imageDataP, chroma.imageLength);
        tex->len[0] = chroma.imageLength;
    }

    auto &luma = imagePointers[crl::multisense::Source_Luma_Rectified_Aux];
    if (luma.imageDataP != nullptr && luma.source == crl::multisense::Source_Luma_Rectified_Aux) {
        tex->data[1] = malloc(luma.imageLength);
        memcpy(tex->data[1], luma.imageDataP, luma.imageLength);
        tex->len[1] = luma.imageLength;
    }

    if (tex->len[0] > 0 && luma.source == crl::multisense::Source_Luma_Rectified_Aux && tex->len[1] > 0 &&
        chroma.source == crl::multisense::Source_Chroma_Rectified_Aux)
        return true;
    else
        return false;

}
 */

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


    assert(tex != nullptr);
    auto src = Utils::stringToDataSource(stringSrc);

    auto header = imagePointersMap[idx][src];
    crl::multisense::DataSource colorSource;
    crl::multisense::DataSource lumaSource;
    if (stringSrc == "Color Aux") {
        colorSource = crl::multisense::Source_Chroma_Aux;
        lumaSource = crl::multisense::Source_Luma_Aux;
    } else if (stringSrc == "Color Rectified Aux") {
        colorSource = crl::multisense::Source_Chroma_Rectified_Aux;
        lumaSource = crl::multisense::Source_Luma_Rectified_Aux;
    }
    auto chroma = imagePointersMap[idx][colorSource];
    auto luma = imagePointersMap[idx][lumaSource];

    switch (tex->type) {
        case AR_COLOR_IMAGE_YUV420:
            if (chroma.imageLength > 0 && (luma.source == crl::multisense::Source_Luma_Rectified_Aux ||
                                           luma.source == crl::multisense::Source_Luma_Aux) &&
                luma.imageLength > 0 &&
                (chroma.source == crl::multisense::Source_Chroma_Rectified_Aux ||
                 chroma.source == crl::multisense::Source_Chroma_Aux)) {

                tex->planar.data[0] = malloc(chroma.imageLength);
                memcpy(tex->planar.data[0], chroma.imageDataP, chroma.imageLength);
                tex->planar.len[0] = chroma.imageLength;
                tex->planar.data[1] = malloc(luma.imageLength);
                memcpy(tex->planar.data[1], luma.imageDataP, luma.imageLength);
                tex->planar.len[1] = luma.imageLength;

                return true;
            } else
                return false;
        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_GRAYSCALE_IMAGE:
        case AR_DISPARITY_IMAGE:
        case AR_POINT_CLOUD:
            if (header.source != src)
                return false;

            if (header.imageDataP != nullptr && header.imageLength != 0 && header.imageLength < 11520000) {
                tex->data = malloc(header.imageLength);
                memcpy(tex->data, header.imageDataP, header.imageLength);
                //tex->data = (void *) header.imageDataP;
                tex->len = header.imageLength;
                return true;
            }
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

    kInverseMatrix =
            glm::mat4(
                    glm::vec4(fy * tx, 0, 0, -fy * cx * tx),
                    glm::vec4(0, fx * tx, 0, -fx * cy * tx),
                    glm::vec4(0, 0, 0, fx * fy * tx),
                    glm::vec4(0, 0, -fy, fy * (cx - cxRight)));

    infoMap[0].kInverseMatrix = glm::transpose(kInverseMatrix);
}


void CRLPhysicalCamera::imuCallback(const crl::multisense::imu::Header &header,
                                    void *userDataP) {
    auto *app = static_cast<CRLPhysicalCamera *>(userDataP);
    std::vector<crl::multisense::imu::Sample>::const_iterator it = header.samples.begin();

    for (; it != header.samples.end(); ++it) {

        const crl::multisense::imu::Sample &s = *it;

        switch (s.type) {
            case crl::multisense::imu::Sample::Type_Accelerometer:
                break;
            case crl::multisense::imu::Sample::Type_Gyroscope:
                break;
            case crl::multisense::imu::Sample::Type_Magnetometer:
                break;
        }

        if (s.type == crl::multisense::imu::Sample::Type_Accelerometer) {
            auto lock = std::scoped_lock<std::mutex>(app->swap_lock);
            float accelX = (s.x);
            float accelY = (s.y);
            float accelZ = (s.z);
            float pitch = 180.0f * atan2(accelX, sqrt(accelY * accelY + accelZ * accelZ)) / (float) M_PI;
            float roll = 180.0f * atan2(accelY, sqrt(accelX * accelX + accelZ * accelZ)) / (float) M_PI;

            app->rotationData.roll = roll;
            app->rotationData.pitch = pitch;

            app->startTimeImu = std::chrono::steady_clock::now();
        }
    }
}

void CRLPhysicalCamera::updateCameraInfo(uint32_t idx) {
    if (crl::multisense::Status_Ok != channelMap[idx]->getImageConfig(infoMap[idx].imgConf)) {
        Log::Logger::getInstance()->info("Failed to update Light config");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getNetworkConfig(infoMap[idx].netConfig)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "netConfig");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getVersionInfo(infoMap[idx].versionInfo)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "versionInfo");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getDeviceInfo(infoMap[idx].devInfo)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "devInfo");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getDeviceModes(infoMap[idx].supportedDeviceModes)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedDeviceModes");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getImageCalibration(infoMap[idx].camCal)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "camCal");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getEnabledStreams(infoMap[idx].supportedSources)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedSources");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getMtu(infoMap[idx].sensorMTU)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "sensorMTU");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[idx]->getLightingConfig(infoMap[idx].lightConf)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "lightConf");
        return;
    }

    if (crl::multisense::Status_Ok != channelMap[idx]->getImageCalibration(infoMap[idx].calibration)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "calibration");
        return;
    }

}


void CRLPhysicalCamera::setGamma(float gamma) {
    crl::multisense::Status status = channelMap[0]->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query gamma configuration");
    }
    infoMap[0].imgConf.setGamma(gamma);
    status = channelMap[0]->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set gamma configuration");
    }

    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setFps(float fps, uint32_t index) {
    crl::multisense::Status status = channelMap[index]->getImageConfig(infoMap[index].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    infoMap[index].imgConf.setFps(fps);

    status = channelMap[index]->setImageConfig(infoMap[index].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setGain(float gain) {
    crl::multisense::Status status = channelMap[0]->getImageConfig(infoMap[0].imgConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }

    infoMap[0].imgConf.setGain(gain);

    status = channelMap[0]->setImageConfig(infoMap[0].imgConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);
}


void CRLPhysicalCamera::setResolution(CRLCameraResolution resolution, uint32_t i) {

    if (resolution == currentResolutionMap[i] || resolution == CRL_RESOLUTION_NONE)
        return;
    uint32_t width = 0, height = 0, depth = 0;
    Utils::cameraResolutionToValue(resolution, &width, &height, &depth);
    if (width == 0 || height == 0 || depth == 0) {
        Log::Logger::getInstance()->error("Resolution mode not supported");
        return;
    }
    int ret = channelMap[i]->getImageConfig(infoMap[i].imgConf);
    if (ret != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->error("failed to get image config");
    }
    if (Utils::valueToCameraResolution(infoMap[i].imgConf.width(), infoMap[i].imgConf.height(),
                                       infoMap[i].imgConf.disparities()) ==
        resolution) {
        currentResolutionMap[i] = resolution;
        Log::Logger::getInstance()->info("Resolution already set to {}x{}x{}", width, height, depth);
        return;
    }

    infoMap[i].imgConf.setResolution(width, height);
    infoMap[i].imgConf.setDisparities(depth);
    ret = channelMap[i]->setImageConfig(infoMap[i].imgConf);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{} on channel {}", width, height, depth, i);
        currentResolutionMap[i] = resolution;
    } else {
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height, depth, ret);
        return;
    }

    this->updateCameraInfo(i);
}

void CRLPhysicalCamera::setExposureParams(ExposureParams p) {

    crl::multisense::Status status = channelMap[0]->getImageConfig(infoMap[0].imgConf);
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
    status = channelMap[0]->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setPostFilterStrength(float filter) {
    crl::multisense::Status status = channelMap[0]->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    infoMap[0].imgConf.setStereoPostFilterStrength(filter);
    status = channelMap[0]->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }
    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setHDR(bool hdr) {
    crl::multisense::Status status = channelMap[0]->getImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query hdr image configuration");
    }

    infoMap[0].imgConf.setHdr(hdr);
    status = channelMap[0]->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set hdr configuration");
    }
    this->updateCameraInfo(0);
}

void CRLPhysicalCamera::setWhiteBalance(WhiteBalanceParams param) {
    crl::multisense::Status status = channelMap[0]->getImageConfig(infoMap[0].imgConf);
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
    status = channelMap[0]->setImageConfig(infoMap[0].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo(0);

}

void CRLPhysicalCamera::setLighting(LightingParams param) {
    crl::multisense::Status status = channelMap[0]->getLightingConfig(info.lightConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }

    if (param.selection != -1) {
        info.lightConf.setDutyCycle(param.selection, param.dutyCycle);
    } else
        info.lightConf.setDutyCycle(param.dutyCycle);

    info.lightConf.setNumberOfPulses(param.numLightPulses);
    info.lightConf.setLedStartupTime(param.startupTime);
    info.lightConf.setFlash(param.flashing);
    status = channelMap[0]->setLightingConfig(info.lightConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }


}

void CRLPhysicalCamera::setMtu(uint32_t mtu, uint32_t channelID) {
    int status = channelMap[channelID]->setMtu(mtu);
    if (status != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Failed to set MTU {}", mtu);
    } else {
        Log::Logger::getInstance()->info("Set MTU to {}", mtu);
    }
}
