//
// Created by magnus on 3/1/22.
//

#include "CRLPhysicalCamera.h"

#include <MultiSense/src/tools/Logger.h>
#include <vulkan/vulkan_core.h>
#include "MultiSense/src/tools/Utils.h"


bool CRLPhysicalCamera::connect(const std::string &ip) {
    if (cameraInterface == nullptr) {
        cameraInterface = crl::multisense::Channel::Create(ip);
        if (cameraInterface != nullptr) {
            updateCameraInfo();
            addCallbacks();

            int mtuSize = 7200;
            int status = cameraInterface->setMtu(mtuSize);
            if (status != crl::multisense::Status_Ok) {
                Log::Logger::getInstance()->info("Failed to set MTU {}", mtuSize);
            } else {
                Log::Logger::getInstance()->info("Set MTU to {}", mtuSize);
            }
            return true;
        }
    }


    return false;
}


bool CRLPhysicalCamera::start(CRLCameraResolution resolution, std::string dataSourceStr) {

    crl::multisense::DataSource source = stringToDataSource(dataSourceStr);
    if (source == false)
        return false;


    // Start stream
    bool status = cameraInterface->startStreams(source);

    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Enabled stream: {}",
                                         dataSourceToString(source).c_str());
        stopForDestruction = false;
        return true;

    } else
        Log::Logger::getInstance()->info("Failed to enable stream: {}  status code {}",
                                         dataSourceToString(source).c_str(), status);
    return false;


}


bool CRLPhysicalCamera::stop(std::string dataSourceStr) {

    if (cameraInterface == nullptr)
        return false;

    crl::multisense::DataSource src = stringToDataSource(dataSourceStr);
    // Check if the stream has been enabled before we attempt to stop it

    /*
    std::vector<uint32_t>::iterator it;
    // Search and stop additional sources
    it = std::find(enabledSources.begin(), enabledSources.end(), crl::multisense::Source_Chroma_Rectified_Aux);
    if (it != enabledSources.end()) {
        src |= crl::multisense::Source_Luma_Rectified_Aux;
    }
    */
    bool status = cameraInterface->stopStreams(src);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Stopped camera stream {}", dataSourceStr.c_str());
        return true;
    } else {
        Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
        return false;
    }
}

bool CRLPhysicalCamera::getCameraStream(ArEngine::YUVTexture *tex) {
    assert(tex != nullptr);
    tex->format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;

    auto chroma = imagePointers[crl::multisense::Source_Chroma_Rectified_Aux];
    if (chroma.imageDataP != nullptr && chroma.source == crl::multisense::Source_Chroma_Rectified_Aux) {
        tex->data[0] = malloc(chroma.imageLength);
        memcpy(tex->data[0], chroma.imageDataP, chroma.imageLength);
        tex->len[0] = chroma.imageLength;
    }

    auto luma = imagePointers[crl::multisense::Source_Luma_Rectified_Aux];
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

bool CRLPhysicalCamera::getCameraStream(std::string stringSrc, ArEngine::TextureData *tex) {
    assert(tex != nullptr);
    auto src = stringToDataSource(stringSrc);

    switch (src) {
        case crl::multisense::Source_Disparity_Left:
            tex->type = AR_DISPARITY_IMAGE;
            break;
        case crl::multisense::Source_Unknown:
            Log::Logger::getInstance()->info("Could not get camera stream. Source unknown");
            return false;
        default:
            tex->type = AR_GRAYSCALE_IMAGE;
    }


    auto header = imagePointers[src];

    // TODO Fix with proper conditions for checking if a frame is good or not
    if (header.source != src)
        return false;

    if (header.imageDataP != nullptr && header.imageLength != 0 && header.imageLength < 11520000) {
        tex->data = malloc(header.imageLength);
        memcpy(tex->data, header.imageDataP, header.imageLength);
        //tex->data = (void *) header.imageDataP;
        tex->len = header.imageLength;
        return true;
    }
    return false;

}


void CRLPhysicalCamera::preparePointCloud(uint32_t width, uint32_t height) {


    crl::multisense::image::Calibration calibration{};
    cameraInterface->getImageCalibration(calibration);

    const double xScale = 1.0 / ((static_cast<double>(info.devInfo.imagerWidth) /
                                  static_cast<double>(width)));

    // From LibMultisenseUtility
    crl::multisense::image::Config c = info.imgConf;
    const double fx = c.fx();
    const double fy = c.fy();
    const double cx = c.cx();
    const double cy = c.cy();
    const double tx = c.tx();
    const double cxRight = calibration.right.P[0][2] * xScale;

    kInverseMatrix =
            glm::mat4(
                    glm::vec4(fy * tx, 0, 0, -fy * cx * tx),
                    glm::vec4(0, fx * tx, 0, -fx * cy * tx),
                    glm::vec4(0, 0, 0, fx * fy * tx),
                    glm::vec4(0, 0, -fy, fy * (cx - cxRight)));

    //kInverseMatrix = glm::transpose(kInverseMatrix); // TODO uncomment here and remove in shader code

    info.kInverseMatrix = kInverseMatrix;
    //info.kInverseMatrix = Q;
    /*
   kInverseMatrix = glm::mat4(glm::vec4(c.fy() * c.tx(), 0, 0, -c.fy() * c.cx() * c.tx()),
                  glm::vec4(0, c.fx() * c.tx(), 0, -c.fx() * c.cy() * c.tx()),
                  glm::vec4(0, 0, 0, c.fx() * c.fy() * c.tx()),
                  glm::vec4(0, 0, -c.fx(), c.fy() * 1));

                      kInverseMatrix =
              glm::mat4(
                      glm::vec4(1/c.fx(), 0, -(c.cx()*c.fx())/(c.fx() * c.fy()), 0),
                      glm::vec4(0, 1/c.fy(), -c.cy() / c.fy(), 0),
                      glm::vec4(0, 0,  1, 0),
                      glm::vec4(0, 0, 0, 1));
  */
    // Load calibration data
}

void CRLPhysicalCamera::updateCameraInfo() {
    cameraInterface->getImageConfig(info.imgConf);
    cameraInterface->getNetworkConfig(info.netConfig);
    cameraInterface->getVersionInfo(info.versionInfo);
    cameraInterface->getDeviceInfo(info.devInfo);
    cameraInterface->getDeviceModes(info.supportedDeviceModes);
    cameraInterface->getImageCalibration(info.camCal);
    cameraInterface->getEnabledStreams(info.supportedSources);
    cameraInterface->getMtu(info.sensorMTU);
}

// Copied from opengl multisense-viewer example
void CRLPhysicalCamera::streamCallback(const crl::multisense::image::Header &image) {

    auto &buf = buffers_[image.source];

    // TODO: make this a method of the BufferPair or something
    std::scoped_lock lock(buf.swap_lock);

    if (buf.inactiveCBBuf != nullptr)  // initial state
    {
        cameraInterface->releaseCallbackBuffer(buf.inactiveCBBuf);
    }


    if (image.imageDataP == nullptr) {
        Log::Logger::getInstance()->info("Image from camera was empty");
    } else
        imagePointers[image.source] = image;

    buf.inactiveCBBuf = cameraInterface->reserveCallbackBuffer();
    buf.inactive = image;
}

void CRLPhysicalCamera::imageCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto cam = reinterpret_cast<CRLPhysicalCamera *>(userDataP);


    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span =
            std::chrono::duration_cast<std::chrono::duration<float>>(time - cam->startTime);

    //Log::Logger::getInstance()->info("Source id: {} time since last call: {}s",header.source, time_span.count());

    cam->startTime = std::chrono::steady_clock::now();

    if (!cam->stopForDestruction)
        cam->streamCallback(header);
}


void CRLPhysicalCamera::addCallbacks() {
    for (auto e: info.supportedDeviceModes)
        info.supportedSources |= e.supportedDataSources;

    // reserve double_buffers for each stream
    uint_fast8_t num_sources = 0;
    crl::multisense::DataSource d = info.supportedSources;
    while (d) {
        num_sources += (d & 1);
        d >>= 1;
    }

    // --- initializing our callback buffers ---
    std::size_t bufSize = 1024 * 1024 * 10;  // 10mb for every image, like in LibMultiSense
    for (int i = 0;
         i < (num_sources * 2 + 1); ++i) // double-buffering for each stream, plus one for handling if those are full
    {
        info.rawImages.push_back(new uint8_t[bufSize]);
    }

    // use these buffers instead of the default
    cameraInterface->setLargeBuffers(info.rawImages, bufSize);

    // finally, add our callback
    if (cameraInterface->addIsolatedCallback(imageCallback, info.supportedSources, this) !=
        crl::multisense::Status_Ok) {
        std::cerr << "Adding callback failed!\n";
    }

    /*
if (cameraInterface->addIsolatedCallback(imageCallback, crl::multisense::Source_All, this) !=
    crl::multisense::Status_Ok) {
    std::cerr << "Adding callback failed!\n";
}
     */

}

CRLBaseInterface::CameraInfo CRLPhysicalCamera::getCameraInfo() {
    return info;
}

void CRLPhysicalCamera::setGamma(float gamma) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to query image configuration");
    }
    //
    // Modify image configuration parameters
    // Here we increase the frame rate to 30 FPS
    info.imgConf.setGamma(gamma);
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setImageConfig(info.imgConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setFps(float fps) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to query image configuration");
    }
    //
    // Modify image configuration parameters
    // Here we increase the frame rate to 30 FPS
    info.imgConf.setFps(fps);
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setImageConfig(info.imgConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setGain(float gain) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to query image configuration");
    }
    //
    // Modify image configuration parameters
    // Here we increase the frame rate to 30 FPS
    info.imgConf.setGain(gain);
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setImageConfig(info.imgConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to set image configuration");
    }

    this->updateCameraInfo();
}


void CRLPhysicalCamera::setResolution(uint32_t width, uint32_t height, uint32_t depth = 64) {

    crl::multisense::image::Config cfg;
    int ret = cameraInterface->getImageConfig(cfg);
    if (ret != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->error("failed to get image config");
    }
    cfg.setResolution(width, height);
    cfg.setDisparities(depth);

    ret = cameraInterface->setImageConfig(cfg);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{}", width, height, depth);
    } else
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height, depth, ret);
    this->updateCameraInfo();


    crl::multisense::lighting::Config c;
}

void CRLPhysicalCamera::setResolution(CRLCameraResolution resolution) {

    if (resolution == currentResolution)
        return;

    uint32_t width, height, depth;
    Utils::cameraResolutionToValue(resolution, &width, &height, &depth);

    crl::multisense::image::Config cfg;
    int ret = cameraInterface->getImageConfig(cfg);
    if (ret != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->error("failed to get image config");
    }
    cfg.setResolution(width, height);
    cfg.setDisparities(depth);

    ret = cameraInterface->setImageConfig(cfg);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{}", width, height, depth);
        currentResolution = resolution;
    } else
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height, depth, ret);
    this->updateCameraInfo();


    crl::multisense::lighting::Config c;
}

void CRLPhysicalCamera::setExposure(uint32_t exp) {
    crl::multisense::image::Config cfg = info.imgConf;

    cfg.setExposure(exp);
    int ret = cameraInterface->setImageConfig(cfg);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set exposure to {}", exp);
    } else
        Log::Logger::getInstance()->info("failed setting exposure to {}", exp);

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setExposureParams(ExposureParams p) {

    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to query image configuration");
    }
    //
    // Modify image configuration parameters
    // Here we increase the frame rate to 30 FPS
    if (p.autoExposure) {
        info.imgConf.setAutoExposure(p.autoExposure);
        info.imgConf.setAutoExposureMax(p.autoExposureMax);
        info.imgConf.setAutoExposureDecay(p.autoExposureDecay);
        info.imgConf.setAutoExposureTargetIntensity(p.autoExposureTargetIntensity);
        info.imgConf.setAutoExposureThresh(p.autoExposureThresh);
    } else {
        info.imgConf.setAutoExposure(p.autoExposure);
        info.imgConf.setExposure(p.exposure);
    }

    info.imgConf.setExposureSource(p.exposureSource);
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setImageConfig(info.imgConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setPostFilterStrength(float filter) {

    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to query image configuration");
    }
    //
    // Modify image configuration parameters
    // Here we increase the frame rate to 30 FPS
    info.imgConf.setStereoPostFilterStrength(filter);
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setImageConfig(info.imgConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setWhiteBalance(WhiteBalanceParams param) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to query image configuration");
    }
    //
    // Modify image configuration parameters
    // Here we increase the frame rate to 30 FPS
    if (param.autoWhiteBalance) {
        info.imgConf.setAutoWhiteBalance(param.autoWhiteBalance);
        info.imgConf.setAutoWhiteBalanceThresh(param.autoWhiteBalanceThresh);
        info.imgConf.setAutoWhiteBalanceDecay(param.autoWhiteBalanceDecay);

    } else {
        info.imgConf.setAutoWhiteBalance(param.autoWhiteBalance);
        info.imgConf.setWhiteBalance(param.whiteBalanceRed, param.whiteBalanceBlue);
    }
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setImageConfig(info.imgConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to set image configuration");
    }

    this->updateCameraInfo();

}


std::string CRLPhysicalCamera::dataSourceToString(crl::multisense::DataSource d) {
    switch (d) {
        case crl::multisense::Source_Raw_Left:
            return "Raw Left";
        case crl::multisense::Source_Raw_Right:
            return "Raw Right";
        case crl::multisense::Source_Luma_Left:
            return "Luma Left";
        case crl::multisense::Source_Luma_Right:
            return "Luma Right";
        case crl::multisense::Source_Luma_Rectified_Left:
            return "Luma Rectified Left";
        case crl::multisense::Source_Luma_Rectified_Right:
            return "Luma Rectified Right";
        case crl::multisense::Source_Chroma_Left:
            return "Color Left";
        case crl::multisense::Source_Chroma_Right:
            return "Source Color Right";
        case crl::multisense::Source_Disparity_Left:
            return "Disparity Left";
        case crl::multisense::Source_Disparity_Cost:
            return "Disparity Cost";
        case crl::multisense::Source_Jpeg_Left:
            return "Jpeg Left";
        case crl::multisense::Source_Rgb_Left:
            return "Source Rgb Left";
        case crl::multisense::Source_Lidar_Scan:
            return "Source Lidar Scan";
        case crl::multisense::Source_Raw_Aux:
            return "Raw Aux";
        case crl::multisense::Source_Luma_Aux:
            return "Luma Aux";
        case crl::multisense::Source_Luma_Rectified_Aux:
            return "Luma Rectified Aux";
        case crl::multisense::Source_Chroma_Aux:
            return "Color Aux";
        case crl::multisense::Source_Chroma_Rectified_Aux:
            return "Color Rectified Aux";
        case crl::multisense::Source_Disparity_Aux:
            return "Disparity Aux";
        default:
            return "Unknown";
    }
}

crl::multisense::DataSource CRLPhysicalCamera::stringToDataSource(const std::string &d) {
    if (d == "Raw Left") return crl::multisense::Source_Raw_Left;
    if (d == "Raw Right") return crl::multisense::Source_Raw_Right;
    if (d == "Luma Left") return crl::multisense::Source_Luma_Left;
    if (d == "Luma Right") return crl::multisense::Source_Luma_Right;
    if (d == "Luma Rectified Left") return crl::multisense::Source_Luma_Rectified_Left;
    if (d == "Luma Rectified Right") return crl::multisense::Source_Luma_Rectified_Right;
    if (d == "Color Left") return crl::multisense::Source_Chroma_Left;
    if (d == "Source Color Right") return crl::multisense::Source_Chroma_Right;
    if (d == "Disparity Left") return crl::multisense::Source_Disparity_Left;
    if (d == "Disparity Cost") return crl::multisense::Source_Disparity_Cost;
    if (d == "Jpeg Left") return crl::multisense::Source_Jpeg_Left;
    if (d == "Source Rgb Left") return crl::multisense::Source_Rgb_Left;
    if (d == "Source Lidar Scan") return crl::multisense::Source_Lidar_Scan;
    if (d == "Raw Aux") return crl::multisense::Source_Raw_Aux;
    if (d == "Luma Aux") return crl::multisense::Source_Luma_Aux;
    if (d == "Luma Rectified Aux") return crl::multisense::Source_Luma_Rectified_Aux;
    if (d == "Color Aux") return crl::multisense::Source_Chroma_Aux;
    if (d == "Color Rectified Aux")
        return crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Luma_Rectified_Aux;;
    if (d == "Disparity Aux") return crl::multisense::Source_Disparity_Aux;
    if (d == "Color + Luma Rectified Aux")
        return crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Luma_Rectified_Aux;
    if (d == "All") return crl::multisense::Source_All;
    return false;
}
