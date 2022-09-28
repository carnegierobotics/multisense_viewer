//
// Created by magnus on 3/1/22.
//

#include "CRLPhysicalCamera.h"

#include <vulkan/vulkan_core.h>
#include "MultiSense/src/Tools/Utils.h"

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

            // Start some timers
            callbackTime = std::chrono::steady_clock::now();
            startTime = std::chrono::steady_clock::now();
            startTimeImu = std::chrono::steady_clock::now();

            return true;
        }
    }
    return false;
}


bool CRLPhysicalCamera::start(CRLCameraResolution resolution, std::string dataSourceStr) {
    crl::multisense::DataSource source = Utils::stringToDataSource(dataSourceStr);
    if (source == false)
        return false;

    // Start stream
    int32_t status = cameraInterface->startStreams(source);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Enabled stream: {}",
                                         Utils::dataSourceToString(source).c_str());
        stopForDestruction = false;
        return true;
    } else
        Log::Logger::getInstance()->info("Failed to flashing stream: {}  status code {}",
                                         Utils::dataSourceToString(source).c_str(), status);

    return false;
}

bool CRLPhysicalCamera::start(crl::multisense::DataSource source) {
    // Start stream
    int32_t status = cameraInterface->startStreams(source);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Enabled stream: {}",
                                         Utils::dataSourceToString(source).c_str());
        stopForDestruction = false;
        return true;
    } else
        Log::Logger::getInstance()->info("Failed to flashing stream: {}  status code {}",
                                         Utils::dataSourceToString(source).c_str(), status);
    return false;
}


bool CRLPhysicalCamera::stop(std::string dataSourceStr) {
    if (cameraInterface == nullptr)
        return false;

    crl::multisense::DataSource src = Utils::stringToDataSource(dataSourceStr);

    bool status = cameraInterface->stopStreams(src);
    if (status == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Stopped camera stream {}", dataSourceStr.c_str());
        return true;
    } else {
        Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
        return false;
    }
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

    cam->callbackTime = std::chrono::steady_clock::now();
    cam->startTime = std::chrono::steady_clock::now();


    if (!cam->stopForDestruction)
        cam->streamCallback(header);

}


void CRLPhysicalCamera::addCallbacks() {
    for (const auto &e: info.supportedDeviceModes)
        info.supportedSources |= e.supportedDataSources;

    // reserve double_buffers for each stream
    uint_fast8_t num_sources = 0;
    crl::multisense::DataSource d = info.supportedSources;
    while (d) {
        num_sources += (d & 1);
        d >>= 1;
    }

    // --- initializing our callback buffers ---
    std::size_t bufSize = (size_t) 1024 * 1024 * 10;  // 10mb for every image, like in LibMultiSense
    for (int i = 0;
         i < (num_sources * 2 + 1); ++i) // double-buffering for each stream, plus one for handling if those are full
    {
        info.rawImages.push_back(new uint8_t[bufSize]);
    }

    // use these buffers instead of the default
    cameraInterface->setLargeBuffers(info.rawImages, static_cast<uint32_t>(bufSize));

    // finally, add our callback
    cameraInterface->addIsolatedCallback(imuCallback, this);

    if (cameraInterface->addIsolatedCallback(imageCallback, info.supportedSources, this) !=
        crl::multisense::Status_Ok) {
        std::cerr << "Adding callback failed!\n";
    }
}

CRLBaseInterface::CameraInfo CRLPhysicalCamera::getCameraInfo() {
    return info;
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
    }


}


bool CRLPhysicalCamera::getCameraStream(VkRender::YUVTexture *tex) {
    assert(tex != nullptr);
    tex->format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;

    auto &chroma = imagePointers[crl::multisense::Source_Chroma_Rectified_Aux];
    if (chroma.imageDataP != nullptr && chroma.source == crl::multisense::Source_Chroma_Rectified_Aux) {
        tex->data[0] = malloc(chroma.imageLength);
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

bool CRLPhysicalCamera::getCameraStream(std::string stringSrc, VkRender::TextureData *tex) {

    auto time = std::chrono::steady_clock::now();
    std::chrono::duration<float> time_span =
            std::chrono::duration_cast<std::chrono::duration<float>>(time - startTime);
    if (time_span.count() > 1) {
        std::chrono::duration<float> callbackTimeSpan =
                std::chrono::duration_cast<std::chrono::duration<float>>(time - callbackTime);

        Log::Logger::getInstance()->info(
                "Requesting image from camera stream. But it is {} seconds since the image stream callback was called.",
                callbackTimeSpan.count());
        startTime = std::chrono::steady_clock::now();
    }


    assert(tex != nullptr);
    auto src = Utils::stringToDataSource(stringSrc);

    auto header = imagePointers[src];
    crl::multisense::DataSource colorSource;
    crl::multisense::DataSource lumaSource;
    if (stringSrc == "Color Aux") {
        colorSource = crl::multisense::Source_Chroma_Aux;
        lumaSource = crl::multisense::Source_Luma_Aux;
    } else if (stringSrc == "Color Rectified Aux") {
        colorSource = crl::multisense::Source_Chroma_Rectified_Aux;
        lumaSource = crl::multisense::Source_Luma_Rectified_Aux;
    }
    auto chroma = imagePointers[colorSource];
    auto luma = imagePointers[lumaSource];

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
    // TODO Fix with proper conditions for checking if a frame is good or not
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


void CRLPhysicalCamera::imuCallback(const crl::multisense::imu::Header &header,
                                    void *userDataP) {
    auto* app = static_cast<CRLPhysicalCamera*>(userDataP);
    std::vector<crl::multisense::imu::Sample>::const_iterator it = header.samples.begin();

    for (; it != header.samples.end(); ++it) {

        const crl::multisense::imu::Sample &s = *it;

        switch (s.type) {
            case crl::multisense::imu::Sample::Type_Accelerometer:
                break;
            case crl::multisense::imu::Sample::Type_Gyroscope:
                // rot x y z
                //printf("Gyro: xyz (%f, %f, %f)\n", s.x, s.y, s.z);
                break;
            case crl::multisense::imu::Sample::Type_Magnetometer:
                break;
        }

        if (s.type == crl::multisense::imu::Sample::Type_Accelerometer) {
            auto lock = std::scoped_lock<std::mutex>(app->swap_lock);
            float accelX = (s.x);
            float accelY = (s.y);
            float accelZ = (s.z);
            float pitch = 180.0f * atan2(accelX, sqrt(accelY*accelY + accelZ*accelZ))/(float) M_PI;
            float roll = 180.0f * atan2(accelY, sqrt(accelX*accelX + accelZ*accelZ))/(float) M_PI;

            app->rotationData.roll = roll;
            app->rotationData.pitch = pitch;

            app->startTimeImu = std::chrono::steady_clock::now();
        }
    }

    /*
         if (-1 == sequence)
        sequence = header.sequence;
    else if ((sequence + 1) != header.sequence) {
        const int32_t d = static_cast<int32_t> (header.sequence - (sequence + 1));
        dropped += d;
    }

    sequence = header.sequence;
    messages ++;
     */
}

void CRLPhysicalCamera::updateCameraInfo() {
    if (crl::multisense::Status_Ok != cameraInterface->getImageConfig(info.imgConf))
        Log::Logger::getInstance()->info("Failed to update Light config");
    if (crl::multisense::Status_Ok != cameraInterface->getNetworkConfig(info.netConfig))
        Log::Logger::getInstance()->info("Failed to update '{}'", "netConfig");
    if (crl::multisense::Status_Ok != cameraInterface->getVersionInfo(info.versionInfo))
        Log::Logger::getInstance()->info("Failed to update '{}'", "versionInfo");
    if (crl::multisense::Status_Ok != cameraInterface->getDeviceInfo(info.devInfo))
        Log::Logger::getInstance()->info("Failed to update '{}'", "devInfo");
    if (crl::multisense::Status_Ok != cameraInterface->getDeviceModes(info.supportedDeviceModes))
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedDeviceModes");
    if (crl::multisense::Status_Ok != cameraInterface->getImageCalibration(info.camCal))
        Log::Logger::getInstance()->info("Failed to update '{}'", "camCal");
    if (crl::multisense::Status_Ok != cameraInterface->getEnabledStreams(info.supportedSources))
        Log::Logger::getInstance()->info("Failed to update '{}'", "supportedSources");
    if (crl::multisense::Status_Ok != cameraInterface->getMtu(info.sensorMTU))
        Log::Logger::getInstance()->info("Failed to update '{}'", "sensorMTU");
    if (crl::multisense::Status_Ok != cameraInterface->getLightingConfig(info.lightConf))
        Log::Logger::getInstance()->info("Failed to update '{}'", "lightConf");
}


void CRLPhysicalCamera::setGamma(float gamma) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query gamma configuration");
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
        Log::Logger::getInstance()->info("Unable to set gamma configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setFps(float fps) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
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
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setGain(float gain) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
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
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo();
}


void CRLPhysicalCamera::setResolution(CRLCameraResolution resolution) {

    if (resolution == currentResolution)
        return;

    uint32_t width = 0, height = 0, depth = 0;
    Utils::cameraResolutionToValue(resolution, &width, &height, &depth);
    if (width == 0 || height == 0 || depth == 0) {
        Log::Logger::getInstance()->error("Resolution mode not supported");
        return;
    }

    int ret = cameraInterface->getImageConfig(info.imgConf);
    if (ret != crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->error("failed to get image config");
    }

    if (Utils::valueToCameraResolution(info.imgConf.width(), info.imgConf.height(), info.imgConf.disparities()) ==
        resolution) {
        currentResolution = resolution;
        Log::Logger::getInstance()->info("Resolution already set to {}x{}x{}", width, height, depth);
        return;
    }

    info.imgConf.setResolution(width, height);
    info.imgConf.setDisparities(depth);

    ret = cameraInterface->setImageConfig(info.imgConf);
    if (ret == crl::multisense::Status_Ok) {
        Log::Logger::getInstance()->info("Set resolution to {}x{}x{}", width, height, depth);
        currentResolution = resolution;
    } else
        Log::Logger::getInstance()->info("Failed setting resolution to {}x{}x{}. Error: {}", width, height, depth, ret);
    this->updateCameraInfo();
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
        Log::Logger::getInstance()->info("Unable to query image configuration");
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
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setPostFilterStrength(float filter) {

    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
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
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo();
}

void CRLPhysicalCamera::setWhiteBalance(WhiteBalanceParams param) {
    crl::multisense::Status status = cameraInterface->getImageConfig(info.imgConf);
    //
    // Check to see if the configuration query succeeded
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
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
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }

    this->updateCameraInfo();

}

void CRLPhysicalCamera::setLighting(LightingParams param) {
    crl::multisense::Status status = cameraInterface->getLightingConfig(info.lightConf);

    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to query image configuration");
    }
    //

    if (param.selection != -1) {
        info.lightConf.setDutyCycle(param.selection, param.dutyCycle);
    } else
        info.lightConf.setDutyCycle(param.dutyCycle);

    info.lightConf.setNumberOfPulses(param.numLightPulses);
    info.lightConf.setLedStartupTime(param.startupTime);
    info.lightConf.setFlash(param.flashing);
    //
    // Send the new image configuration to the sensor
    status = cameraInterface->setLightingConfig(info.lightConf);
    //
    // Check to see if the configuration was successfully received by the
    // sensor
    if (crl::multisense::Status_Ok != status) {
        Log::Logger::getInstance()->info("Unable to set image configuration");
    }


}