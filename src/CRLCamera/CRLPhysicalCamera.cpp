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


std::vector<crl::multisense::RemoteHeadChannel> CRLPhysicalCamera::connect(const std::string &ip, bool isRemoteHead) {
    std::vector<crl::multisense::RemoteHeadChannel> indices;
    // If RemoteHead then attempt to connect 4 LibMultiSense channels
    // else create only one and place it at 0th index.
    for (crl::multisense::RemoteHeadChannel i = 0; i <= (crl::multisense::Remote_Head_3); ++i) {
        channelMap[i] = isRemoteHead ? std::make_unique<ChannelWrapper>(ip, i) : std::make_unique<ChannelWrapper>(ip);
        if (channelMap[i].get()->ptr() != nullptr) {
            updateCameraInfo(i);
            setMtu(7200, i);
            addCallbacks(i);
            indices.emplace_back(i);
        }

        if (!isRemoteHead)
            break;
    }
    return indices;
}

bool CRLPhysicalCamera::start(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
    crl::multisense::DataSource source = Utils::stringToDataSource(dataSourceStr);
    if (source == false)
        return false;
    // Start stream
    int32_t status = channelMap[channelID]->ptr()->startStreams(source);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Enabled stream: {} on channel {}",
                                         Utils::dataSourceToString(source).c_str(), channelID);
        return true;
    } else
        Log::Logger::getInstance()->info("Failed to flashing stream: {}  status code {}",
                                         Utils::dataSourceToString(source).c_str(), status);
    return false;
}

bool CRLPhysicalCamera::stop(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
    if (channelMap[channelID] == nullptr)
        return false;
    crl::multisense::DataSource src = Utils::stringToDataSource(dataSourceStr);
    if (!src){
        Log::Logger::getInstance()->info("Failed to recognize '{}' source", dataSourceStr.c_str());
        return false;
    }
    bool status = channelMap[channelID]->ptr()->stopStreams(src);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Stopped camera stream {} on channel {}", dataSourceStr.c_str(), channelID);
        return true;
    } else {
        Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
        return false;
    }
}

void CRLPhysicalCamera::remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto p = reinterpret_cast<ChannelWrapper *>(userDataP);
    //Log::Logger::getInstance()->info("Received image {}x{} on channel {} with source {}", header.width, header.height, p->imageBuffer->id, header.source);
    p->imageBuffer->updateImageBuffer(std::make_shared<ImageBufferWrapper>(p->ptr(), header));
}

void CRLPhysicalCamera::addCallbacks(crl::multisense::RemoteHeadChannel channelID) {
    for (const auto &e: infoMap[channelID].supportedDeviceModes)
        infoMap[channelID].supportedSources |= e.supportedDataSources;
    // reserve double_buffers for each stream
    uint_fast8_t num_sources = 0;
    crl::multisense::DataSource d = infoMap[channelID].supportedSources;
    while (d) {
        num_sources += (d & 1);
        d >>= 1;
    }
    if (channelMap[channelID]->ptr()->addIsolatedCallback(remoteHeadCallback, infoMap[channelID].supportedSources,
        channelMap[channelID].get()) ==
        crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Added callback for channel {}", channelID);
    }
    else 
        Log::Logger::getInstance()->info("Failed to add callback for channel {}", channelID);
}

CRLPhysicalCamera::CameraInfo CRLPhysicalCamera::getCameraInfo(crl::multisense::RemoteHeadChannel idx) {
    return infoMap[idx];
}

bool CRLPhysicalCamera::getCameraStream(std::string stringSrc, VkRender::TextureData *tex,
                                        crl::multisense::RemoteHeadChannel channelID) {
    if (channelMap[channelID] == nullptr)
        return false;

    auto src = Utils::stringToDataSource(stringSrc);
    crl::multisense::DataSource colorSource;
    crl::multisense::DataSource lumaSource;
    std::shared_ptr<ImageBufferWrapper> header;
    std::shared_ptr<ImageBufferWrapper> headerTwo;
    if (stringSrc == "Color Aux") {
        colorSource = crl::multisense::Source_Chroma_Aux;
        lumaSource = crl::multisense::Source_Luma_Aux;
        header = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, lumaSource);
        headerTwo = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, colorSource);
        if (headerTwo == nullptr)
            return false;

    } else if (stringSrc == "Color Rectified Aux") {
        colorSource = crl::multisense::Source_Chroma_Rectified_Aux;
        lumaSource = crl::multisense::Source_Luma_Rectified_Aux;
        header = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, lumaSource);
        headerTwo = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, colorSource);
        if (headerTwo == nullptr)
            return false;
    } else {
        header = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, src);
    }

    if (header == nullptr)
        return false;

    switch (tex->m_Type) {
        case AR_COLOR_IMAGE_YUV420:
            if ((header->data().source | headerTwo->data().source) != src || tex->m_Width != header->data().width ||
                tex->m_Height != header->data().height)
                return false;
            tex->m_Id = header->data().frameId;
            tex->m_Id2 = headerTwo->data().frameId;

            std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
            std::memcpy(tex->data2, headerTwo->data().imageDataP, headerTwo->data().imageLength);
            return true;

        case AR_DISPARITY_IMAGE:
            if (header->data().bitsPerPixel != 16) {
                std::cerr << "Unsupported disparity pixel depth" << std::endl;
                break;
            }
        case AR_GRAYSCALE_IMAGE:
        case AR_POINT_CLOUD:
            if (header->data().source != src || tex->m_Width != header->data().width ||
                tex->m_Height != header->data().height)
                return false;
            tex->m_Id = header->data().frameId;
            std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
            return true;
        default:
            Log::Logger::getInstance()->info("This texture type is not supported {}", (int) tex->m_Type);
            break;
    }
    return false;
}


void CRLPhysicalCamera::preparePointCloud(uint32_t width, uint32_t height) {
    const double xScale = 1.0 / ((static_cast<double>(infoMap[0].devInfo.imagerWidth) /
                                  static_cast<double>(width)));
    // From LibMultisenseUtility
    crl::multisense::image::Config c = infoMap[0].imgConf;
    const float& fx = c.fx();
    const float& fy = c.fy();
    const float& cx = c.cx();
    const float& cy = c.cy();
    const float& tx = c.tx();
    const float cxRight = (float) infoMap[0].calibration.right.P[0][2] * xScale;

    // glm::mat4 indexing
    // [column][row]
    // Inserted values row by row
    glm::mat4 Q(0.0f);
    Q[0][0] = fy * tx;
    Q[3][0] = -fy * cx * tx;
    Q[1][1] = fx * tx;
    Q[3][1] = -fx * cy * tx;
    Q[3][2] = fx * fy * tx;
    Q[2][3] = -fy;
    Q[3][3] = fy * (cx - cxRight);
    // keep as is
    infoMap[0].kInverseMatrix = Q;
}


void CRLPhysicalCamera::updateCameraInfo(crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to update Light config");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getNetworkConfig(infoMap[channelID].netConfig)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "netConfig");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getVersionInfo(infoMap[channelID].versionInfo)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "versionInfo");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getDeviceInfo(infoMap[channelID].devInfo)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "devInfo");
        return;
    }
    if (crl::multisense::Status_Ok !=
        channelMap[channelID]->ptr()->getDeviceModes(infoMap[channelID].supportedDeviceModes)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedDeviceModes");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].camCal)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "camCal");
        return;
    }
    if (crl::multisense::Status_Ok !=
        channelMap[channelID]->ptr()->getEnabledStreams(infoMap[channelID].supportedSources)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedSources");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getMtu(infoMap[channelID].sensorMTU)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "sensorMTU");
        return;
    }
    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "lightConf");
        return;
    }

    if (crl::multisense::Status_Ok !=
        channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].calibration)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "calibration");
        return;
    }
}


bool CRLPhysicalCamera::setGamma(float gamma, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);
    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query gamma configuration");
        return false;
    }
    infoMap[channelID].imgConf.setGamma(gamma);
    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set gamma configuration");
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to verify Gamma");
        return false;
    }
    return true;
}

bool CRLPhysicalCamera::setFps(float fps, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);
    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
        return false;
    }
    infoMap[channelID].imgConf.setFps(fps);

    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to verify fps");
        return false;
    }
    return true;
}

bool CRLPhysicalCamera::setGain(float gain, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);
    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
        return false;
    }

    infoMap[channelID].imgConf.setGain(gain);
    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));


    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to update Gain");
        return false;
    }
    return true;
}


bool CRLPhysicalCamera::setResolution(CRLCameraResolution resolution, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    if (resolution == currentResolutionMap[channelID] || resolution == CRL_RESOLUTION_NONE)
        return false;
    uint32_t width = 0, height = 0, depth = 0;
    Utils::cameraResolutionToValue(resolution, &width, &height, &depth);
    if (width == 0 || height == 0 || depth == 0) {
        Log::Logger::getInstance()->error("Resolution mode not supported");
        return false;
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
        return false;
    }

    infoMap[channelID].imgConf.setResolution(width, height);
    infoMap[channelID].imgConf.setDisparities(depth);
    ret = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{} on channel {}", width, height, depth, channelID);
        currentResolutionMap[channelID] = resolution;
    } else {
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height, depth, ret);
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));


    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to verify resolution");
        return false;
    }

    return true;
}

bool CRLPhysicalCamera::setExposureParams(ExposureParams p, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
        return false;
    }
    if (p.autoExposure) {
        infoMap[channelID].imgConf.setAutoExposure(p.autoExposure);
        infoMap[channelID].imgConf.setAutoExposureMax(p.autoExposureMax);
        infoMap[channelID].imgConf.setAutoExposureDecay(p.autoExposureDecay);
        infoMap[channelID].imgConf.setAutoExposureTargetIntensity(p.autoExposureTargetIntensity);
        infoMap[channelID].imgConf.setAutoExposureThresh(p.autoExposureThresh);
    } else {
        infoMap[channelID].imgConf.setAutoExposure(p.autoExposure);
        infoMap[channelID].imgConf.setExposure(p.exposure);
    }

    infoMap[channelID].imgConf.setExposureSource(p.exposureSource);
    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to verify Exposure params");
        return false;
    }
    return true;
}

bool CRLPhysicalCamera::setPostFilterStrength(float filter, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
        return false;
    }
    infoMap[channelID].imgConf.setStereoPostFilterStrength(filter);
    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to verified post filter strength");
        return false;
    }
    return true;
}

bool CRLPhysicalCamera::setHDR(bool hdr, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query hdr image configuration");
        return false;
    }

    infoMap[channelID].imgConf.setHdr(hdr);
    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set hdr configuration");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to verifiy HDR");
        return false;
    }
    return true;
}

bool CRLPhysicalCamera::setWhiteBalance(WhiteBalanceParams param, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
        return false;
    }
    if (param.autoWhiteBalance) {
        infoMap[channelID].imgConf.setAutoWhiteBalance(param.autoWhiteBalance);
        infoMap[channelID].imgConf.setAutoWhiteBalanceThresh(param.autoWhiteBalanceThresh);
        infoMap[channelID].imgConf.setAutoWhiteBalanceDecay(param.autoWhiteBalanceDecay);

    } else {
        infoMap[channelID].imgConf.setAutoWhiteBalance(param.autoWhiteBalance);
        infoMap[channelID].imgConf.setWhiteBalance(param.whiteBalanceRed, param.whiteBalanceBlue);
    }
    status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
        Log::Logger::getInstance()->info("Failed to update white balance");
        return false;
    }

    return true;
}

bool CRLPhysicalCamera::setLighting(LightingParams param, crl::multisense::RemoteHeadChannel channelID) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    crl::multisense::Status status = channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
        return false;
    }

    if (param.selection != -1) {
        infoMap[channelID].lightConf.setDutyCycle(param.selection, param.dutyCycle);
    } else
        infoMap[channelID].lightConf.setDutyCycle(param.dutyCycle);

    infoMap[channelID].lightConf.setNumberOfPulses(param.numLightPulses);
    infoMap[channelID].lightConf.setStartupTime(param.startupTime);
    infoMap[channelID].lightConf.setFlash(param.flashing);
    status = channelMap[channelID]->ptr()->setLightingConfig(infoMap[channelID].lightConf);
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf)) {
        Log::Logger::getInstance()->info("Failed to update '{}' ", "Lightning");
        return false;
    }

    return true;
}

bool CRLPhysicalCamera::setMtu(uint32_t mtu, crl::multisense::RemoteHeadChannel id) {
    std::scoped_lock<std::mutex> lock(setCameraDataMutex);

    int status = channelMap[id]->ptr()->setMtu((int32_t) mtu);
    if (status != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Failed to set MTU {}", mtu);
        return false;
    } else {
        Log::Logger::getInstance()->info("Set MTU to {}", mtu);
        return true;
    }

    if (crl::multisense::Status_Ok != channelMap[id]->ptr()->getMtu(infoMap[id].sensorMTU)) {
        Log::Logger::getInstance()->info("Failed to update '{}'", "sensorMTU");
        return false;
    }
    return true;
}
