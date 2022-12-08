//
// Created by magnus on 3/1/22.
//
#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif
#include <vulkan/vulkan_core.h>

#include "Viewer/CRLCamera/CRLPhysicalCamera.h"
#include "Viewer/Core/Definitions.h"
#include "Viewer/CRLCamera/CalibrationYaml.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender::MultiSense {

    std::vector<crl::multisense::RemoteHeadChannel>
    CRLPhysicalCamera::connect(const VkRender::Device *dev, bool isRemoteHead, const std::string &ifName) {
        std::vector<crl::multisense::RemoteHeadChannel> indices;
        // If RemoteHead then attempt to connect 4 LibMultiSense channels
        // else create only one and place it at 0th index.
        for (crl::multisense::RemoteHeadChannel i = 0; i <= (crl::multisense::Remote_Head_3); ++i) {
            if (dev->interruptConnection) {
                indices.clear();
                return indices;
            }
            channelMap[i] = isRemoteHead ? std::make_unique<ChannelWrapper>(dev->IP, i, ifName)
                                         : std::make_unique<ChannelWrapper>(dev->IP, crl::multisense::Remote_Head_VPB,
                                                                            ifName);

            if (channelMap[i].get()->ptr() != nullptr) {
                updateCameraInfo(i);
                addCallbacks(i);
                indices.emplace_back(i);
            }
            if (!isRemoteHead)
                break;
        }

        if (dev->interruptConnection) {
            indices.clear();
            return indices;
        }

        if (isRemoteHead) {
            channelMap[crl::multisense::Remote_Head_VPB] = std::make_unique<ChannelWrapper>(dev->IP,
                                                                                            crl::multisense::Remote_Head_VPB,
                                                                                            ifName);
            if (channelMap[crl::multisense::Remote_Head_VPB].get()->ptr() != nullptr)
                setMtu(7200, crl::multisense::Remote_Head_VPB);
        } else {
            if (channelMap[0].get()->ptr() != nullptr)
                setMtu(7200, 0);

        }
        return indices;
    }

    bool CRLPhysicalCamera::start(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
        crl::multisense::DataSource source = Utils::stringToDataSource(dataSourceStr);
        if (source == false)
            return false;
        // Start stream
        crl::multisense::Status status = channelMap[channelID]->ptr()->startStreams(source);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Started stream: {} on channel {}",
                                             Utils::dataSourceToString(source).c_str(), channelID);
            return true;
        } else
            Log::Logger::getInstance()->info("Failed to start stream: {}  status code {}",
                                             Utils::dataSourceToString(source).c_str(), status);
        return false;
    }

    bool CRLPhysicalCamera::stop(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
        if (channelMap[channelID] == nullptr)
            return false;
        crl::multisense::DataSource src = Utils::stringToDataSource(dataSourceStr);
        if (!src) {
            Log::Logger::getInstance()->info("Failed to recognize '{}' source", dataSourceStr.c_str());
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->stopStreams(src);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Stopped camera stream {} on channel {}", dataSourceStr.c_str(),
                                             channelID);
            return true;
        } else {
            Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
            return false;
        }
    }

    void CRLPhysicalCamera::remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP) {
        auto p = reinterpret_cast<ChannelWrapper *>(userDataP);
        //Log::Logger::getInstance()->info("Received m_Image {}x{} on channel {} with source {}", header.m_Width, header.m_Height, p->imageBuffer->id, header.source);
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
        if (channelMap[channelID]->ptr()->addIsolatedCallback(remoteHeadCallback,
                                                              infoMap[channelID].supportedSources,
                                                              channelMap[channelID].get()) ==
            crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Added callback for channel {}", channelID);
        } else
            Log::Logger::getInstance()->info("Failed to add callback for channel {}", channelID);
    }

    CRLPhysicalCamera::CameraInfo CRLPhysicalCamera::getCameraInfo(crl::multisense::RemoteHeadChannel idx) {
        return infoMap[idx];
    }

    bool CRLPhysicalCamera::getCameraStream(const std::string &stringSrc, VkRender::TextureData *tex,
                                            crl::multisense::RemoteHeadChannel channelID) {
        if (channelMap[channelID] == nullptr) {
            Log::Logger::getInstance()->error("Channel not connected {}", channelID);
            return false;
        }

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
            if (headerTwo == nullptr) {
                return false;
            }

        } else if (stringSrc == "Color Rectified Aux") {
            colorSource = crl::multisense::Source_Chroma_Rectified_Aux;
            lumaSource = crl::multisense::Source_Luma_Rectified_Aux;
            header = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, lumaSource);
            headerTwo = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, colorSource);
            if (headerTwo == nullptr) {
                return false;
            }
        } else {
            header = channelMap[channelID]->imageBuffer->getImageBuffer(channelID, src);
        }

        if (header == nullptr) {
            return false;
        }

        switch (tex->m_Type) {
            case CRL_COLOR_IMAGE_YUV420: {
                if (headerTwo->data().source != src ||
                    tex->m_Width != header->data().width ||
                    tex->m_Height < header->data().height) {
                    Log::Logger::getInstance()->error("In getCameraStream: Color Source and dimensions did not match expected values");
                    return false;
                }
                tex->m_Id = static_cast<uint32_t>(header->data().frameId);
                tex->m_Id2 = static_cast<uint32_t>(headerTwo->data().frameId);

                // These should be the same height. This can happen with the Color_Aux source when switching between max and h res.
                if (header->data().height == headerTwo->data().height) {
                    Log::Logger::getInstance()->error("In getCameraStream: Color Source and dimensions does not have matching height");
                    return false;
                }

                std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
                // if data3 is not null we are using a manual ycbcr sampler. Therefore do a strided copy
                if (tex->data3 != nullptr) {
                    size_t channelSize = headerTwo->data().imageLength / 2;
                    auto *p = (uint16_t *) headerTwo->data().imageDataP;
                    for (size_t i = 0; i < channelSize; i++) {
                        tex->data2[i] = ((uint16_t) p[i] >> 0) & 0xFF;
                        // shift by 0 not needed, of course, just stylistic
                        tex->data3[i] = ((uint16_t) p[i] >> 8) & 0xFF;
                    }
                } else
                    std::memcpy(tex->data2, headerTwo->data().imageDataP, headerTwo->data().imageLength);

                // Copy extra zeros to the bottom row if heights does not match
                if (tex->m_Height != header->data().height) {
                    uint32_t diff = tex->m_Height - header->data().height;
                    std::memset(tex->data + header->data().imageLength, 0x00, diff * tex->m_Width);

                    if (tex->data3 != nullptr) {
                        size_t diff2 = (size_t) (tex->m_Height / 2) - headerTwo->data().height;
                        std::memset(tex->data2 + (headerTwo->data().imageLength/ 2), 0x00, diff2 * tex->m_Width);
                        std::memset(tex->data3 + (headerTwo->data().imageLength / 2), 0x00, diff2 * tex->m_Width);

                    } else {
                        uint32_t diff2 = tex->m_Height / 2 - headerTwo->data().height;
                        std::memset(tex->data2 + headerTwo->data().imageLength, 0x00, diff2 * tex->m_Width);
                    }
                }
            }
                return true;

            case CRL_DISPARITY_IMAGE:
                DISABLE_WARNING_PUSH
                DISABLE_WARNING_IMPLICIT_FALLTHROUGH
                if (header->data().bitsPerPixel != 16) {
                    Log::Logger::getInstance()->error("In getCameraStream: Unsupported disparity pixel depth");
                    return false;
                }DISABLE_WARNING_POP
            case CRL_GRAYSCALE_IMAGE:
            case CRL_POINT_CLOUD:
                if (header->data().source != src || tex->m_Width != header->data()
                        .width || tex->m_Height < header->data().height) {
                    Log::Logger::getInstance()->error("In getCameraStream: Monochrome source and dimensions did not match expected values");
                    return false;
                }
                tex->m_Id = static_cast<uint32_t>(header->data().frameId);
                std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
// Copy extra zeros (black pixels) to the bottom row if heights does not match
                if (tex->m_Height != header->data().height) {
                    uint32_t diff = tex->m_Height - header->data().height;
                    std::memset(tex->data + header->data().imageLength, 0x00, diff * tex->m_Width);
                }

                return true;
            default:
                Log::Logger::getInstance()->info("This texture type is not supported {}", (int) tex->m_Type);
                break;
        }
        return false;
    }


    void CRLPhysicalCamera::preparePointCloud(uint32_t width, crl::multisense::RemoteHeadChannel channelID) {
        const float xScale = 1.0f / ((static_cast<float>(infoMap[channelID].devInfo.imagerWidth) /
                                      static_cast<float>(width)));
        // From LibMultisenseUtility
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        crl::multisense::image::Config c = infoMap[channelID].imgConf;
        const float &fx = c.fx();
        const float &fy = c.fy();
        const float &cx = c.cx();
        const float &cy = c.cy();
        const float &tx = c.tx();
        const float cxRight = (float) infoMap[channelID].calibration.right.P[0][2] * xScale;

        // glm::mat4 indexing
        // [column][row]
        // Inserted values col by col
        glm::mat4 Q(0.0f);
        Q[0][0] = fy * tx;
        Q[1][1] = fx * tx;
        Q[2][3] = -fy;
        Q[3][0] = -fy * cx * tx;
        Q[3][1] = -fx * cy * tx;
        Q[3][2] = fx * fy * tx;
        Q[3][3] = fy * (cx - cxRight);
        // keep as is
        infoMap[channelID].QMat = Q;
    }


    void CRLPhysicalCamera::updateCameraInfo(crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->error("Failed to update Light config");
            return;
        }
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getNetworkConfig(infoMap[channelID].netConfig)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "netConfig");
            return;
        }
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getVersionInfo(infoMap[channelID].versionInfo)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "versionInfo");
            return;
        }
        if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getDeviceInfo(infoMap[channelID].devInfo)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "devInfo");
            return;
        }
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getDeviceModes(infoMap[channelID].supportedDeviceModes)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "supportedDeviceModes");
            return;
        }
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getEnabledStreams(infoMap[channelID].supportedSources)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "supportedSources");
            return;
        }
        if (crl::multisense::Status_Ok != channelMap[channelID]->ptr()->getMtu(infoMap[channelID].sensorMTU)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "sensorMTU");
            return;
        }
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "lightConf");
            return;
        }

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].calibration)) {
            Log::Logger::getInstance()->error("Failed to update '{}'", "calibration");
            return;
        }
    }


    bool CRLPhysicalCamera::setGamma(float gamma, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set gamma on a channel that was not connected, Channel {}", channelID);
            return false;
        }
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
        } else {
            Log::Logger::getInstance()->info("Set Gamma on channel {}", channelID);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to verify Gamma");
            return false;
        }
        return true;
    }

    bool CRLPhysicalCamera::setFps(float fps, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set fps on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query m_Image configuration");
            return false;
        }
        infoMap[channelID].imgConf.setFps(fps);

        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set m_Image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set framerate on channel {}", channelID);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        /*
        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to verify fps");
            return false;
        }
         */

        // Since we can only have one framerate among several remote head update all the infoMaps.
        for (const auto &channel: channelMap) {
            // don't bother for the VPB
            if (channel.first == crl::multisense::Remote_Head_VPB || channel.second->ptr() == nullptr)
                continue;
            if (crl::multisense::Status_Ok != channel.second->ptr()->getImageConfig(infoMap[channel.first].imgConf)) {
                Log::Logger::getInstance()->info("Failed to verify fps");
                return false;
            }
        }
        return true;
    }

    bool CRLPhysicalCamera::setGain(float gain, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);

        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set gain on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query image configuration");
            return false;
        }

        infoMap[channelID].imgConf.setGain(gain);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set gain on channel {}", channelID);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));


        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to update Gain");
            return false;
        }
        return true;
    }


    bool
    CRLPhysicalCamera::setResolution(CRLCameraResolution resolution, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID] == nullptr){
            Log::Logger::getInstance()->error("Attempted to set resolution on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        if (resolution == CRL_RESOLUTION_NONE) {
            Log::Logger::getInstance()->info("Resolution is not specified {}", (int) resolution);
            return false;
        }
        uint32_t width = 0, height = 0, depth = 0;
        Utils::cameraResolutionToValue(resolution, &width, &height, &depth);
        if (width == 0 || height == 0 || depth == 0) {
            Log::Logger::getInstance()->error("Resolution mode not supported");
            return false;
        }
        int ret = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (ret != crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->error("failed to get m_Image config");
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
            Log::Logger::getInstance()->info("Set resolution to {}x{}x{} on channel {}", width, height, depth,
                                             channelID);
            currentResolutionMap[channelID] = resolution;
        } else {
            Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height,
                                             depth, ret);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));


        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to verify resolution");
            return false;
        }

        return true;
    }

    bool CRLPhysicalCamera::setExposureParams(ExposureParams p, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set exposure on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to query exposure configuration");
            return false;
        }
        if (p.autoExposure) {
            infoMap[channelID].imgConf.setAutoExposure(p.autoExposure);
            infoMap[channelID].imgConf.setAutoExposureMax(p.autoExposureMax);
            infoMap[channelID].imgConf.setAutoExposureDecay(p.autoExposureDecay);
            infoMap[channelID].imgConf.setAutoExposureTargetIntensity(p.autoExposureTargetIntensity);
            infoMap[channelID].imgConf.setAutoExposureThresh(p.autoExposureThresh);
            infoMap[channelID].imgConf.setAutoExposureRoi(p.autoExposureRoiX, p.autoExposureRoiY,
                                                          p.autoExposureRoiWidth,
                                                          p.autoExposureRoiHeight);

        } else {
            infoMap[channelID].imgConf.setAutoExposure(p.autoExposure);
            infoMap[channelID].imgConf.setExposure(p.exposure);
        }

        infoMap[channelID].imgConf.setExposureSource(p.exposureSource);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to set exposure configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set exposure on channel {}", channelID);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->error("Failed to verify Exposure params");
            return false;
        }
        return true;
    }

    bool CRLPhysicalCamera::setPostFilterStrength(float filter, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set post filter strength on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to query m_Image configuration");
            return false;
        }
        infoMap[channelID].imgConf.setStereoPostFilterStrength(filter);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to set m_Image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Successfully set stereo post filter strength to {} on channel {}", filter,
                                             channelID);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        // Since we can only have one stereo setting among several remote head update all the infoMaps.
        for (const auto &channel: channelMap) {
            // don't bother for the VPB
            if (channel.first == crl::multisense::Remote_Head_VPB || channel.second->ptr() == nullptr)
                continue;
            if (crl::multisense::Status_Ok != channel.second->ptr()->getImageConfig(infoMap[channel.first].imgConf)) {
                Log::Logger::getInstance()->info("Failed to verify fps");
                return false;
            }
        }
        return true;
    }

    bool CRLPhysicalCamera::setHDR(bool hdr, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set hdr on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query hdr m_Image configuration");
            return false;
        }

        infoMap[channelID].imgConf.setHdr(hdr);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set hdr configuration");
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to verifiy HDR");
            return false;
        }
        return true;
    }

    bool
    CRLPhysicalCamera::setWhiteBalance(WhiteBalanceParams param, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set white balance on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        //
        // Check to see if the configuration query succeeded
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query m_Image configuration");
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
            Log::Logger::getInstance()->info("Unable to set m_Image configuration");
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to update white balance");
            return false;
        }

        return true;
    }

    bool CRLPhysicalCamera::setLighting(LightingParams param, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set light configuration on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getLightingConfig(
                infoMap[channelID].lightConf);

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query m_Image configuration");
            return false;
        }

        if (param.selection != -1) {
            infoMap[channelID].lightConf.setDutyCycle(param.selection, param.dutyCycle);
        } else
            infoMap[channelID].lightConf.setDutyCycle(param.dutyCycle);

        infoMap[channelID].lightConf.setNumberOfPulses((uint32_t)param.numLightPulses * 1000);  // convert from ms to us and from float to uin32_t
        infoMap[channelID].lightConf.setStartupTime((uint32_t)(param.startupTime * 1000)); // convert from ms to us and from float to uin32_t
        infoMap[channelID].lightConf.setFlash(param.flashing);
        status = channelMap[channelID]->ptr()->setLightingConfig(infoMap[channelID].lightConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set m_Image configuration");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf)) {
            Log::Logger::getInstance()->info("Failed to update '{}' ", "Lightning");
            return false;
        }

        return true;
    }

    bool CRLPhysicalCamera::setMtu(uint32_t mtu, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set mtu on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        int status = channelMap[channelID]->ptr()->setMtu((int32_t) mtu);
        if (status != crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Failed to set MTU {}", mtu);
            return false;
        } else {
            Log::Logger::getInstance()->info("Set MTU to {}", mtu);
            return true;
        }
    }

    bool CRLPhysicalCamera::getStatus(crl::multisense::RemoteHeadChannel channelID,
                                      crl::multisense::system::StatusMessage *msg) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr)
            return false;

        crl::multisense::Status status = channelMap[channelID]->ptr()->getDeviceStatus(*msg);
        return status == crl::multisense::Status_Ok;
    }

    bool CRLPhysicalCamera::setSensorCalibration(const std::string &intrinsicsFile, const std::string &extrinsicsFile,
                                                 crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to set sensor calibration on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageCalibration(
                infoMap[channelID].calibration);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query calibration");
            return false;
        }
        bool hasAuxCamera = infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM;
        std::ifstream inFile, exFile;
        std::map<std::string, std::vector<float> > data;
        inFile.open(intrinsicsFile.c_str());
        if (!inFile) {
            Log::Logger::getInstance()->error("Failed to open intrinsics file: {}.", intrinsicsFile);
            return false;
        }
        parseYaml(inFile, data);
        inFile.close();
        if (data["M1"].size() != 3 * 3 ||
            (data["D1"].size() != 5 && data["D1"].size() != 8) ||
            data["M2"].size() != 3 * 3 ||
            (data["D2"].size() != 5 && data["D2"].size() != 8) ||
            (hasAuxCamera && data["M3"].size() != 3 * 3) ||
            (hasAuxCamera && data["D3"].size() != 5 && data["D3"].size() != 8)) {
            Log::Logger::getInstance()->error("Intrinsics file error: {}. File not complete", intrinsicsFile);
            return false;
        }
        exFile.open(extrinsicsFile.c_str());
        if (!exFile) {
            Log::Logger::getInstance()->error("Failed to open extrinsics file: {}.", extrinsicsFile);
            return false;
        }
        parseYaml(exFile, data);
        exFile.close();
        if (data["R1"].size() != 3 * 3 ||
            data["P1"].size() != 3 * 4 ||
            data["R2"].size() != 3 * 3 ||
            data["P2"].size() != 3 * 4 ||
            (hasAuxCamera && (data["R3"].size() != 3 * 3 || data["P3"].size() != 3 * 4))) {
            Log::Logger::getInstance()->error("Extrinsics file error: {}. File not complete", intrinsicsFile);
            return false;
        }

        crl::multisense::image::Calibration calibration{};

        memcpy(&calibration.left.M[0][0], &data["M1"].front(), data["M1"].size() * sizeof(float));
        memset(&calibration.left.D[0], 0, sizeof(calibration.left.D));
        memcpy(&calibration.left.D[0], &data["D1"].front(), data["D1"].size() * sizeof(float));
        memcpy(&calibration.left.R[0][0], &data["R1"].front(), data["R1"].size() * sizeof(float));
        memcpy(&calibration.left.P[0][0], &data["P1"].front(), data["P1"].size() * sizeof(float));

        memcpy(&calibration.right.M[0][0], &data["M2"].front(), data["M2"].size() * sizeof(float));
        memset(&calibration.right.D[0], 0, sizeof(calibration.right.D));
        memcpy(&calibration.right.D[0], &data["D2"].front(), data["D2"].size() * sizeof(float));
        memcpy(&calibration.right.R[0][0], &data["R2"].front(), data["R2"].size() * sizeof(float));
        memcpy(&calibration.right.P[0][0], &data["P2"].front(), data["P2"].size() * sizeof(float));

        if (hasAuxCamera) {
            memcpy(&calibration.aux.M[0][0], &data["M3"].front(), data["M3"].size() * sizeof(float));
            memset(&calibration.aux.D[0], 0, sizeof(calibration.aux.D));
            memcpy(&calibration.aux.D[0], &data["D3"].front(), data["D3"].size() * sizeof(float));
            memcpy(&calibration.aux.R[0][0], &data["R3"].front(), data["R3"].size() * sizeof(float));
            memcpy(&calibration.aux.P[0][0], &data["P3"].front(), data["P3"].size() * sizeof(float));
        }


        status = channelMap[channelID]->ptr()->setImageCalibration(calibration);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to set calibration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Successfully set new calibration");

        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].calibration)) {
            Log::Logger::getInstance()->info("Failed to query image calibration");
            return false;
        }
        return true;
    }


    std::ostream &
    CRLPhysicalCamera::writeImageIntrinics(std::ostream &stream, crl::multisense::image::Calibration const &calibration,
                                           bool hasAuxCamera) {
        stream << "%YAML:1.0\n";
        writeMatrix(stream, "M1", 3, 3, &calibration.left.M[0][0]);
        writeMatrix(stream, "D1", 1, 8, &calibration.left.D[0]);
        writeMatrix(stream, "M2", 3, 3, &calibration.right.M[0][0]);
        writeMatrix(stream, "D2", 1, 8, &calibration.right.D[0]);

        if (hasAuxCamera) {
            writeMatrix(stream, "M3", 3, 3, &calibration.aux.M[0][0]);
            writeMatrix(stream, "D3", 1, 8, &calibration.aux.D[0]);
        }
        return stream;
    }


    std::ostream &
    CRLPhysicalCamera::writeImageExtrinics(std::ostream &stream, crl::multisense::image::Calibration const &calibration,
                                           bool hasAuxCamera) {
        stream << "%YAML:1.0\n";
        writeMatrix(stream, "R1", 3, 3, &calibration.left.R[0][0]);
        writeMatrix(stream, "P1", 3, 4, &calibration.left.P[0][0]);
        writeMatrix(stream, "R2", 3, 3, &calibration.right.R[0][0]);
        writeMatrix(stream, "P2", 3, 4, &calibration.right.P[0][0]);

        if (hasAuxCamera) {
            writeMatrix(stream, "R3", 3, 3, &calibration.aux.R[0][0]);
            writeMatrix(stream, "P3", 3, 4, &calibration.aux.P[0][0]);
        }
        return stream;
    }


    bool CRLPhysicalCamera::saveSensorCalibration(const std::string &savePath,
                                                  crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr){
            Log::Logger::getInstance()->error("Attempted to save calibration from a channel that was not connected, Channel {}", channelID);
            return false;
        }

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].calibration)) {
            Log::Logger::getInstance()->info("Failed to query image calibration");
            return false;
        }

        bool hasAuxCamera = infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM;
        std::ofstream inFile, exFile;
        std::string intrinsicsFile = savePath + "/intrinsics.yml";
        std::string extrinsicsFile = savePath + "/extrinsics.yml";

        inFile.open(intrinsicsFile.c_str(), std::ios_base::out | std::ios_base::trunc);

        if (!inFile) {
            Log::Logger::getInstance()->error("Failed to open Intrinsics file for writing!");

            return false;
        }

        exFile.open(extrinsicsFile.c_str(), std::ios_base::out | std::ios_base::trunc);

        if (!exFile) {
            Log::Logger::getInstance()->error("Failed to open Extrinsics file for writing!");
            return false;
        }

        writeImageIntrinics(inFile, infoMap[channelID].calibration, hasAuxCamera);
        writeImageExtrinics(exFile, infoMap[channelID].calibration, hasAuxCamera);
        inFile.flush();
        exFile.flush();

        Log::Logger::getInstance()->info("Saved camera calibration to file. HasAux: {}", hasAuxCamera);
        return true;
    }

}