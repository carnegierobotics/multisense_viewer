//
// Created by magnus on 6/26/24.
//

#include "GigEVisionConnector.h"

#include "Viewer/Tools/Logger.h"

namespace VkRender::MultiSense {

    static std::mutex s_imageMutex;
    static std::condition_variable s_conditionVariable;
    static GigEVisionConnector::Image s_image;
    static std::unordered_map<int, image_data> s_imageObjects;

    void GigEVisionConnector::connect(std::string ip, std::string ifName) {

        int32_t numDevices = m_gigEv->enumerate();
        if (numDevices > 0) {
            m_gigEv->connect(numDevices - 1);
            m_gigEv->set_fps(15);
            //cam_device.enable_stream_source(GEV_SOURCE_LUMA_LEFT, stream_callback);
            //m_gigEv->enable_stream_source(GEV_SOURCE_LUMA_RIGHT, GigEVisionConnector::streamCallback);
            //m_gigEv->enable_stream_source(GEV_SOURCE_DISPARITY_LEFT, GigEVisionConnector::streamCallback);
            m_gigEv->enable_stream_source(GEV_SOURCE_LUMA_LEFT, GigEVisionConnector::streamCallback);
            m_gigEv->acquire();
            s_image.img = static_cast<uint8_t *>(std::malloc(960 * 600));
            s_image.imageSize = 960 * 600;
            s_image.width = 960;
            s_image.height = 600;
        }
    }

    void GigEVisionConnector::disconnect() {
        m_gigEv->stop_acquisition();
        m_gigEv->disconnect();

    }

    void GigEVisionConnector::update() {


        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_lastEnumerateTime).count();

        if (elapsed >= 5) {
            // Run the enumerate function every 5 seconds
            //int32_t numDevices = m_gigEv->enumerate();
            // Reset the last enumeration time to now
            m_lastEnumerateTime = now;
        }





    }

    void GigEVisionConnector::setup() {
        int32_t numDevices = m_gigEv->enumerate();
        if (numDevices > 0) {
            m_gigEv->connect(numDevices - 1);
            m_gigEv->set_fps(15);
            //cam_device.enable_stream_source(GEV_SOURCE_LUMA_LEFT, stream_callback);
            //m_gigEv->enable_stream_source(GEV_SOURCE_LUMA_RIGHT, GigEVisionConnector::streamCallback);
            //m_gigEv->enable_stream_source(GEV_SOURCE_DISPARITY_LEFT, GigEVisionConnector::streamCallback);
            m_gigEv->enable_stream_source(GEV_SOURCE_LUMA_LEFT, GigEVisionConnector::streamCallback);
            m_gigEv->acquire();
            s_image.img = static_cast<uint8_t *>(std::malloc(960 * 600));
            s_image.imageSize = 960 * 600;
            s_image.width = 960;
            s_image.height = 600;
        }
    }

    uint8_t *GigEVisionConnector::getImage() {
        std::unique_lock<std::mutex> lk(s_imageMutex); // Lock the mutex
        auto timeout = std::chrono::milliseconds(0);
        // Use wait_for to periodically check if the condition is met
        if (s_conditionVariable.wait_for(lk, timeout, [] {
            // Check if the image is ready
            return !s_imageObjects.empty();
        })) {
            // Image is ready, process it
            auto imageInfo = s_imageObjects.begin()->second;


            if (s_imageObjects[imageInfo.source_port].pixel_format == GEV_PIXFMT_MONO8) {
                // we have an image;
                Log::Logger::getInstance()->info("We have an image: {}",
                                                 s_imageObjects[imageInfo.source_port].block_id);
            }

            return s_imageObjects[imageInfo.source_port].image_buf.data();

        } else {
            // Timeout occurred, you can do something else or retry
            // Perform other tasks or simply retry
        }
        return nullptr;
    }

    void GigEVisionConnector::streamCallback(image_data info, uint8_t *imageBuffer) {
        {
            (void) imageBuffer; // Unused (for now)
            std::lock_guard lk(s_imageMutex);
            s_imageObjects[info.source_port] = info;

            std::memcpy(s_image.img, imageBuffer, s_image.imageSize);

        }

        s_conditionVariable.notify_one();
    }


}