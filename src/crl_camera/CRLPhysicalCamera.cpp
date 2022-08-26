//
// Created by magnus on 3/1/22.
//


#include <MultiSense/src/tools/Logger.h>
#include <vulkan/vulkan_core.h>
#include "CRLPhysicalCamera.h"


bool CRLPhysicalCamera::connect(const std::string &ip) {
    if (cameraInterface == nullptr) {
        cameraInterface = crl::multisense::Channel::Create(ip);
        if (cameraInterface != nullptr) {
            updateCameraInfo();
            addCallbacks();

            /*
            crl::multisense::system::NetworkConfig config;
            bool status = cameraInterface->getNetworkConfig(config); // TODO Move and error check this line. Failed on Windows if Jumbo frames is disabled on ethernet device
            if (status != crl::multisense::Status_Ok){
                std::cerr << "Failed to set MTU 7200\n";
            }
             */

            int status = cameraInterface->setMtu(7200);
            if (status != crl::multisense::Status_Ok) {
                std::cerr << "Failed to set MTU 7200\n";
            }
            return true;
        }
    }


    return false;
}


void CRLPhysicalCamera::start(std::string string, std::string dataSourceStr) {


    // Set mode first
    std::string delimiter = "x";

    size_t pos = 0;
    std::string token;
    std::vector<uint32_t> widthHeightDepth;
    while ((pos = string.find(delimiter)) != std::string::npos) {
        token = string.substr(0, pos);
        widthHeightDepth.push_back(std::stoi(token));
        string.erase(0, pos + delimiter.length());
    }
    if (widthHeightDepth.size() != 3) {
        std::cerr << "Select valid resolution\n";
        return;
    }

    // If res changed then set it again.
    if (info.imgConf.disparities() != widthHeightDepth[2]) {
        Log::Logger::getInstance()->info("CRLPhysicalCamera:: Setting resolution {} x {} x {}", widthHeightDepth[0],
                                         widthHeightDepth[1], widthHeightDepth[2]);
        setResolution(widthHeightDepth[0], widthHeightDepth[1], widthHeightDepth[2]);
    }


    crl::multisense::DataSource source = stringToDataSource(dataSourceStr);
    if (source == false)
        return;
    // Check if the stream has already been enabled first
    if (std::find(enabledSources.begin(), enabledSources.end(),
                  source) != enabledSources.end()) {
        return;
    }
    if (dataSourceStr == "Color Rectified Aux") {
        enabledSources.push_back(
                crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Luma_Rectified_Aux);
    } else {
        enabledSources.push_back(source);
    }

    // Start stream
    for (auto src: enabledSources) {
        bool status = cameraInterface->startStreams(src);

        if (status == crl::multisense::Status_Ok)
            Log::Logger::getInstance()->info("CRLPhysicalCamera:: Enabled stream: {}",
                                             dataSourceToString(src).c_str());
        else
            Log::Logger::getInstance()->info("CRLPhysicalCamera:: Failed to enable stream: {}  status code {}",
                                             dataSourceToString(src).c_str(), status);

    }

    for (auto src: enabledSources) {

    }

}


void CRLPhysicalCamera::stop(std::string dataSourceStr) {
    Log::Logger::getInstance()->info("CRLPhysicalCamera:: Stopping camera streams {}", dataSourceStr.c_str());

    if (cameraInterface == nullptr)
        return;
    crl::multisense::DataSource src = stringToDataSource(dataSourceStr);
    // Check if the stream has been enabled before we attempt to stop it
    if (dataSourceStr != "All") {
        if (std::find(enabledSources.begin(), enabledSources.end(),
                      src) == enabledSources.end()) {
            return;
        }
        std::vector<uint32_t>::iterator it;
        it = std::remove(enabledSources.begin(), enabledSources.end(),
                         src);
        enabledSources.erase(it);
    } else {
        enabledSources.clear();
    }
    /*
    std::vector<uint32_t>::iterator it;
    // Search and stop additional sources
    it = std::find(enabledSources.begin(), enabledSources.end(), crl::multisense::Source_Chroma_Rectified_Aux);
    if (it != enabledSources.end()) {
        src |= crl::multisense::Source_Luma_Rectified_Aux;
    }
    */
    bool status = cameraInterface->stopStreams(src);
    Log::Logger::getInstance()->info("CRLPhysicalCamera:: Stopped camera streams {}", dataSourceStr.c_str());
}

bool CRLPhysicalCamera::getCameraStream(ArEngine::YUVTexture *tex) {
    assert(tex != nullptr);
    tex->format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;

    auto chroma = imagePointers[crl::multisense::Source_Chroma_Rectified_Aux];
    if (chroma.imageDataP != nullptr) {
        tex->data[0] = (void *) chroma.imageDataP;
        tex->len[0] = chroma.imageLength;
    }

    auto luma = imagePointers[crl::multisense::Source_Luma_Rectified_Aux];
    if (luma.imageDataP != nullptr) {
        tex->data[1] = (void *) luma.imageDataP;
        tex->len[1] = luma.imageLength;
    }

    if (luma.imageDataP != nullptr && chroma.imageDataP != nullptr)
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
        default:
            tex->type = AR_GRAYSCALE_IMAGE;
    }


    auto header = imagePointers[src];
    if (header.imageDataP != nullptr) {
        tex->data = (void *) header.imageDataP;
        tex->len = header.imageLength;
        return true;
    }
    return false;

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
    if (d == "Color Rectified Aux") return crl::multisense::Source_Chroma_Rectified_Aux;
    if (d == "Disparity Aux") return crl::multisense::Source_Disparity_Aux;
    if (d == "Color + Luma Rectified Aux")
        return crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Luma_Rectified_Aux;
    if (d == "All") return crl::multisense::Source_All;
    return false;
}

void CRLPhysicalCamera::update() {

    for (auto src: enabledSources) {
        if (src == crl::multisense::Source_Disparity_Left) {
            // Reproject camera to 3D
            //stream = &imagePointers[crl::multisense::Source_Disparity_Left];

        }
    }

}

void CRLPhysicalCamera::preparePointCloud(uint32_t width, uint32_t height) {

    crl::multisense::image::Config c = info.imgConf;

    kInverseMatrix =
            glm::mat4(
                    glm::vec4(1 / c.fx(), 0, -(c.cx() * c.fx()) / (c.fx() * c.fy()), 0),
                    glm::vec4(0, 1 / c.fy(), -c.cy() / c.fy(), 0),
                    glm::vec4(0, 0, 1, 0),
                    glm::vec4(0, 0, 0, 1));

    kInverseMatrix = glm::transpose(kInverseMatrix);
    crl::multisense::image::Config params = info.imgConf;
    crl::multisense::image::Calibration cal = info.camCal;

    float dcx = (cal.right.P[0][2] - cal.left.P[0][2]) *
                (1.0f / static_cast<float>(info.devInfo.imagerWidth * params.width()));
    glm::mat4 Q = glm::mat4(glm::vec4(params.fy() * params.tx(), 0, 0, -params.fy() * params.cx() * params.tx()),
                            glm::vec4(0, params.fx() * params.tx(), 0, -params.fx() * params.cy() * params.tx()),
                            glm::vec4(0, 0, 0, params.fx() * params.fy() * params.tx()),
                            glm::vec4(0, 0, -params.fx(), params.fy() * (dcx)));

    start("960 x 600 x 64x", "Disparity Left");

    info.kInverseMatrix = kInverseMatrix;
    //kInverseMatrix = Q;
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
        Log::Logger::getInstance()->error("CRLPhysicalCamera:: failed to get image config");
    }
    cfg.setResolution(width, height);
    cfg.setDisparities(depth);

    ret = cameraInterface->setImageConfig(cfg);
    ret = cameraInterface->setImageConfig(cfg);
    if (ret != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{}", width, height, depth);
    } else
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}", width, height, depth);
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

