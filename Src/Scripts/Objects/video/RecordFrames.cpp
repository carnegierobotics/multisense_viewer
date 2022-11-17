//
// Created by magnus on 10/12/22.
//

#include "RecordFrames.h"
#include "MultiSense/external/TinyTIFF/src/tinytiffwriter.h"

void RecordFrames::setup() {
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    threadPool = std::make_unique<VkRender::ThreadPool>();

}

void RecordFrames::update() {
    if (!saveImage)
        return;

    // For each enabled source in all windows
    for (const auto &src: sources) {
        // For each remote head index
        for (crl::multisense::RemoteHeadChannel remoteIdx = crl::multisense::Remote_Head_0;
             remoteIdx <= crl::multisense::Remote_Head_3; ++remoteIdx) {

            const auto &conf = renderData.crlCamera->get()->getCameraInfo(remoteIdx).imgConf;

            auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(),
                                                               conf.height(), true);


            if (renderData.crlCamera->get()->getCameraStream(src, tex.get(), remoteIdx)) {
                if (src == "Color Aux" || src == "Color Rectified Aux") {
                    if (tex->m_Id != tex->m_Id2)
                        continue;
                }
                if (threadPool->getTaskListSize() < STACK_SIZE_100) {
                    threadPool->Push(saveImageToFile, Utils::CRLSourceToTextureType(src), saveImagePath, src,
                                     remoteIdx, tex, isRemoteHead);
                }
            }
        }

    }
}

void RecordFrames::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != AR_STATE_ACTIVE)
            continue;

        sources.clear();
        for (const auto &window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource))
                sources.emplace_back(window.second.selectedSource);
        }

        saveImagePath = dev.outputSaveFolder;
        saveImage = dev.isRecording;
        isRemoteHead = dev.isRemoteHead;
    }
}


void RecordFrames::saveImageToFile(CRLCameraDataType type, const std::string &path, const std::string &stringSrc,
                                   crl::multisense::RemoteHeadChannel remoteHead,
                                   std::shared_ptr<VkRender::TextureData> &ptr, bool isRemoteHead) {
#ifdef WIN32
    std::string separator = "\\";
#else
    std::string separator = "/";
#endif

    std::string directory, remoteHeadDir;
    directory = std::string(path);
    std::string fileName = std::to_string(ptr->m_Id);
    if (isRemoteHead) {
        remoteHeadDir = path + separator + std::to_string(remoteHead);
        directory = remoteHeadDir + separator + std::string(stringSrc);
    } else
        directory = std::string(path) + separator + stringSrc;

    std::string filePath = directory + std::string(separator);

#ifdef WIN32
    // Create Remote head index directory
    if (isRemoteHead) {
        int check = _mkdir(remoteHeadDir.c_str());
        if (check == 0)
            Log::Logger::getInstance()->info("Created directory {}", remoteHeadDir);

    }
    // Create Source name directory
    int check = _mkdir(directory.c_str());
    if (check == 0)
        Log::Logger::getInstance()->info("Created directory {}", directory);

    if (type == AR_DISPARITY_IMAGE){
        // Create Source name directory
        check = _mkdir((directory + "/png").c_str());
        if (check == 0)
            Log::Logger::getInstance()->info("Created directory {}", directory + "png");

        check = _mkdir((directory + "/tiff").c_str());
        if (check == 0)
            Log::Logger::getInstance()->info("Created directory {}", directory + "tiff");
    }

#else
    // Create Remote head index directory

    if (isRemoteHead) {
        int check = mkdir(remoteHeadDir.c_str(), 0777);
        if (check == 0)
            Log::Logger::getInstance()->info("Created directory {}", remoteHeadDir);
    }

    // Create Source name directory
    int check = mkdir(directory.c_str(), 0777);
    if (check == 0)
        Log::Logger::getInstance()->info("Created directory {}", directory);

    if (type == AR_DISPARITY_IMAGE){
        // Create Source name directory
        check = mkdir((directory + "/png").c_str(), 0777);
        if (check == 0)
            Log::Logger::getInstance()->info("Created directory {}", directory + "png");

        check = mkdir((directory + "/tiff").c_str(), 0777);
        if (check == 0)
            Log::Logger::getInstance()->info("Created directory {}", directory + "tiff");
    }

#endif

    std::string fullPathName = filePath + fileName + ".png";
    // Check if file exists. Otherwise do nothing
    std::ifstream fin(fullPathName.c_str());
    if (fin.good()) {
        fin.close();
        return;
    }
    // Create file


    switch (type) {
        case AR_POINT_CLOUD:
            break;
        case AR_GRAYSCALE_IMAGE:
        {
            std::ofstream output(fullPathName);
            output.close();
            Log::Logger::getInstance()->info("Saving Frame: {} from source: {}", ptr->m_Id, stringSrc);

            stbi_write_png((fullPathName).c_str(), ptr->m_Width, ptr->m_Height, 1, ptr->data,
                           ptr->m_Width);
        }
            break;
        case AR_DISPARITY_IMAGE:
        {
            auto *d = (uint16_t *) ptr->data;
            std::string fullTIFFPath = filePath + "tiff/" + fileName + ".tif";
            std::string fullPngPath = filePath + "png/" + fileName + ".png";
            Log::Logger::getInstance()->info("Saving Frame: {} from source: {}", ptr->m_Id, stringSrc);

            TinyTIFFWriterFile* tif=TinyTIFFWriter_open(fullTIFFPath.c_str(), 16, TinyTIFFWriter_UInt, 1,  ptr->m_Width, ptr->m_Height, TinyTIFFWriter_Greyscale);
            TinyTIFFWriter_writeImage(tif, d);
            TinyTIFFWriter_close(tif);

            std::vector<uint8_t> buf;
            buf.reserve(ptr->m_Width * ptr->m_Height);
            for (size_t i = 0; i < ptr->m_Width * ptr->m_Height; ++i) {
                d[i] /= 16;
                uint8_t lsb = d[i] & 0x000000FF;
                buf.emplace_back(lsb);
            }
            stbi_write_png((fullPngPath).c_str(), ptr->m_Width, ptr->m_Height, 1, buf.data(),
                           ptr->m_Width);
        }
            break;
        case AR_COLOR_IMAGE_YUV420:
            // Normalize ycbcr
        {
            std::ofstream output(fullPathName);
            output.close();
            Log::Logger::getInstance()->info("Saving Frame: {} from source: {}", ptr->m_Id, stringSrc);


            int width = ptr->m_Width;
            int height = ptr->m_Height;

            AVFrame *src;
            src = av_frame_alloc();
            if (!src) {
                Log::Logger::getInstance()->error("Could not allocate video frame");
            }
            src->format = AV_PIX_FMT_YUV420P;
            src->width = width;
            src->height = height;
            int ret = av_image_alloc(src->data, src->linesize, src->width, src->height,
                                     static_cast<AVPixelFormat>(src->format), 32);
            if (ret < 0) {
                Log::Logger::getInstance()->error("Could not allocate raw picture buffer");
            }

            std::memcpy(src->data[0], ptr->data, ptr->m_Len);
            auto *d = (uint16_t *) ptr->data2;
            for (size_t i = 0; i < ptr->m_Height / 2 * ptr->m_Width / 2; ++i) {
                src->data[1][i] = d[i] & 0xff;
                src->data[2][i] = (d[i] >> (8)) & 0xff;
            }

            AVFrame *dst;
            dst = av_frame_alloc();
            if (!dst) {
                Log::Logger::getInstance()->error("Could not allocate video frame");
            }
            dst->format = AV_PIX_FMT_RGB24;
            dst->width = width;
            dst->height = height;
            ret = av_image_alloc(dst->data, dst->linesize, dst->width, dst->height,
                                 static_cast<AVPixelFormat>(dst->format), 8);
            if (ret < 0) {
                Log::Logger::getInstance()->error("Could not allocate raw picture buffer");
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
            stbi_write_png((fullPathName).c_str(), width, height, 3, dst->data[0], dst->linesize[0]);

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
        case AR_COLOR_IMAGE:
            break;
    }
}
