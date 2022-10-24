//
// Created by magnus on 10/12/22.
//

#include "RecordFrames.h"


void RecordFrames::setup() {
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    threadPool = std::make_unique<VkRender::ThreadPool>();

}

void RecordFrames::update() {
    if (!saveImage)
        return;

    for (const auto &src: sources) {
        const auto &conf = renderData.crlCamera->get()->getCameraInfo(remoteHeadIndex).imgConf;
        auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(), conf.height(), true);

        if (renderData.crlCamera->get()->getCameraStream(src, tex.get(), remoteHeadIndex)) {
            if (threadPool->getTaskListSize() < STACK_SIZE_100) {
                Log::Logger::getInstance()->info("Saving image {} from source {}", tex->m_Id, src);
                threadPool->Push(saveImageToFile, Utils::CRLSourceToTextureType(src), saveImagePath, src, tex);
            }
        }

    }
}

void RecordFrames::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        sources.clear();
        for (const auto& window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource))
                sources.emplace_back(window.second.selectedSource);
        }

        saveImage = dev.isRecording;
        saveImagePath = dev.outputSaveFolder;
    }
}



void RecordFrames::saveImageToFile(CRLCameraDataType type, const std::string &path, const std::string& stringSrc, std::shared_ptr<VkRender::TextureData>& ptr) {

    std::string fileName = std::to_string(ptr->m_Id);
    std::string directory = path + "/" + stringSrc;
    std::string filePath = directory + "/";

#ifdef WIN32
    int check = _mkdir(directory.c_str());
#else
    int check = mkdir(directory.c_str(), 0777);
#endif
// check if directory is created or not
    if (!check)
        printf("Directory created\n");

    std::string fullPathName = filePath + fileName;
    switch (type) {
        case AR_POINT_CLOUD:
            break;
        case AR_GRAYSCALE_IMAGE:
            stbi_write_png((fullPathName + ".png").c_str(), ptr->m_Width, ptr->m_Height, 1, ptr->data,
                           ptr->m_Width);
            break;
        case AR_DISPARITY_IMAGE: {
            auto *d = (uint16_t *) ptr->data;
            std::vector<uint8_t> buf;
            buf.reserve(ptr->m_Width * ptr->m_Height);
            for (int i = 0; i < ptr->m_Width * ptr->m_Height; ++i) {
                d[i] /= 16;
                uint8_t lsb = d[i] & 0x000000FF;
                buf.emplace_back(lsb);
            }
            stbi_write_png((fullPathName + ".png").c_str(), ptr->m_Width, ptr->m_Height, 1, buf.data(),
                           ptr->m_Width);
        }
            break;
        case AR_COLOR_IMAGE_YUV420:
            // Normalize ycbcr
        {
            int width = ptr->m_Width;
            int height = ptr->m_Height;

            AVFrame *src;
            src = av_frame_alloc();
            if (!src) {
                fprintf(stderr, "Could not allocate video frame\n");
            }
            src->format = AV_PIX_FMT_YUV420P;
            src->width  = width;
            src->height = height;
            int ret = av_image_alloc(src->data, src->linesize, src->width, src->height,
                                     static_cast<AVPixelFormat>(src->format), 32);
            if (ret < 0) {
                fprintf(stderr, "Could not allocate raw picture buffer\n");
            }

            std::memcpy(src->data[0], ptr->data, ptr->m_Len);
            auto* d = (uint16_t*) ptr->data2;
            for (int i = 0; i < ptr->m_Height / 2 * ptr->m_Width/2; ++i) {
                src->data[1][i] = d[i] & 0xff;
                src->data[2][i] = (d[i] >> (8)) & 0xff;
            }

            AVFrame *dst;
            dst = av_frame_alloc();
            if (!dst) {
                fprintf(stderr, "Could not allocate video frame\n");
            }
            dst->format = AV_PIX_FMT_RGB24;
            dst->width  = width;
            dst->height = height;
            ret = av_image_alloc(dst->data, dst->linesize, dst->width, dst->height,
                                 static_cast<AVPixelFormat>(dst->format), 8);
            if (ret < 0) {
                fprintf(stderr, "Could not allocate raw picture buffer\n");
            }
            SwsContext *conversion = sws_getContext(width,
                                                    height,
                                                    (AVPixelFormat) AV_PIX_FMT_YUV420P,
                                                    width,
                                                    height,
                                                    AV_PIX_FMT_RGB24,
                                                    SWS_FAST_BILINEAR,
                                                    NULL,
                                                    NULL,
                                                    NULL);
            sws_scale(conversion, src->data, src->linesize, 0, height, dst->data, dst->linesize);
            sws_freeContext(conversion);
            stbi_write_png((fullPathName + ".png").c_str(),width, height, 3, dst->data[0], dst->linesize[0]);

            av_freep(&src->data[0]);
            av_frame_free(&src);
            av_freep(&dst->data[0]);
            av_frame_free(&dst);

        }
            break;
        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_CAMERA_IMAGE_NONE:
            break;
    }
}
