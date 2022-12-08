//
// Created by magnus on 10/12/22.
//
#include <TinyTIFF/src/tinytiffwriter.h>

#include "Viewer/Scripts/Objects/Video/RecordFrames.h"

#include <filesystem>

extern "C" {
#include<libavutil/avutil.h>
#include<libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

void RecordFrames::setup() {
    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());
    threadPool = std::make_unique<VkRender::ThreadPool>(3);

}

void RecordFrames::update() {
    if (!saveImage)
        return;

    // For each enabled source in all windows
    for (auto &src: sources) {
        if (src == "Source")
            continue;
        // For each remote head index
        for (crl::multisense::RemoteHeadChannel remoteIdx = crl::multisense::Remote_Head_0;
             remoteIdx <=
             (isRemoteHead ? crl::multisense::Remote_Head_3 : crl::multisense::Remote_Head_0); ++remoteIdx) {
            const auto &conf = renderData.crlCamera->get()->getCameraInfo(remoteIdx).imgConf;
            auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(),
                                                               conf.height(), true);

            if (renderData.crlCamera->get()->getCameraStream(src, tex.get(), remoteIdx)) {
                if (src == "Color Aux" || src == "Color Rectified Aux") {
                    if (tex->m_Id != tex->m_Id2)
                        continue;
                }
                Log::Logger::getInstance()->info("Record queue size: {}", threadPool->getTaskListSize());

                if (threadPool->getTaskListSize() < MAX_IMAGES_IN_QUEUE) {
                    threadPool->Push(saveImageToFile, Utils::CRLSourceToTextureType(src), saveImagePath, src,
                                     remoteIdx, tex, isRemoteHead, compression);
                } else if (threadPool->getTaskListSize() >= MAX_IMAGES_IN_QUEUE && compression == "tiff") {
                    Log::Logger::getInstance()->info("Record image queue is full. Starting to drop frames");

                }
            }
        }

    }
}

void RecordFrames::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const VkRender::Device &dev: uiHandle->devices) {
        if (dev.state != CRL_STATE_ACTIVE)
            continue;

        sources.clear();
        for (const auto &window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource))
                sources.emplace_back(window.second.selectedSource);
        }

        saveImagePath = dev.outputSaveFolder;
        saveImage = dev.isRecording;
        compression = dev.saveImageCompressionMethod;
        isRemoteHead = dev.isRemoteHead;

    }
}

void RecordFrames::saveImageToFile(CRLCameraDataType type, const std::string &path, std::string &stringSrc,
                                   crl::multisense::RemoteHeadChannel remoteHead,
                                   std::shared_ptr<VkRender::TextureData> &ptr, bool isRemoteHead,
                                   const std::string &compression) {
    // Create folders. One for each Source
    std::filesystem::path saveDirectory{};
    std::replace(stringSrc.begin(), stringSrc.end(), ' ', '_');
    if (isRemoteHead) {
        saveDirectory = path + "/Head_" + std::to_string(remoteHead + 1) + "/" + stringSrc + "/";
    } else {
        saveDirectory = path + "/" + stringSrc + "/" + compression + "/";
    }
    std::filesystem::path fileLocation = saveDirectory.string() + std::to_string(ptr->m_Id) + "." + compression;
    // Dont overwrite already saved images
    if (std::filesystem::exists(fileLocation))
        return;
    // Create folders if no exists
    if (!std::filesystem::is_directory(saveDirectory) ||
        !std::filesystem::exists(saveDirectory)) { // Check if src folder exists
        std::filesystem::create_directories(saveDirectory); // create src folder
    }
    // Create file
    if (compression == "tiff") {
        switch (type) {
            case CRL_POINT_CLOUD:
                break;
            case CRL_GRAYSCALE_IMAGE: {
                auto *d = ptr->data;
                TinyTIFFWriterFile *tif = TinyTIFFWriter_open(fileLocation.c_str(), 8, TinyTIFFWriter_UInt, 1,
                                                              ptr->m_Width, ptr->m_Height, TinyTIFFWriter_Greyscale);
                TinyTIFFWriter_writeImage(tif, d);
                TinyTIFFWriter_close(tif);
            }
                break;
            case CRL_DISPARITY_IMAGE: {
                auto *d =  ptr->data;
                TinyTIFFWriterFile *tif = TinyTIFFWriter_open(fileLocation.c_str(), 16, TinyTIFFWriter_UInt, 1,
                                                              ptr->m_Width, ptr->m_Height, TinyTIFFWriter_Greyscale);
                TinyTIFFWriter_writeImage(tif, d);
                TinyTIFFWriter_close(tif);
            }
                break;
            case CRL_COLOR_IMAGE_YUV420:
                // Normalize ycbcr
            {
                std::ofstream outputStream((fileLocation).c_str(),
                                           std::ios::out | std::ios::binary);
                if (!outputStream.good()) {
                    Log::Logger::getInstance()->error("Failed top open file {} for writing", fileLocation.string());
                    break;
                }
                const uint32_t imageSize = ptr->m_Height * ptr->m_Width * 3;
                outputStream << "P6\n"
                             << ptr->m_Width << " " << ptr->m_Height << "\n"
                             << 0xFF << "\n";
                outputStream.write(reinterpret_cast<const char *>(ptr->data), imageSize);
                outputStream.close();
                break;
            }
            case CRL_YUV_PLANAR_FRAME:
            case CRL_CAMERA_IMAGE_NONE:
            case CRL_COLOR_IMAGE:
                break;
        }
    } else if (compression == "png") {
        switch (type) {
            case CRL_POINT_CLOUD:
                break;
            case CRL_GRAYSCALE_IMAGE: {

                stbi_write_png((fileLocation).c_str(), ptr->m_Width, ptr->m_Height, 1, ptr->data,
                               ptr->m_Width);
            }
                break;
            case CRL_DISPARITY_IMAGE: {
                auto *d = reinterpret_cast<uint16_t *>( ptr->data);
                std::vector<uint8_t> buf;
                buf.reserve(ptr->m_Width * ptr->m_Height);
                for (size_t i = 0; i < ptr->m_Width * ptr->m_Height; ++i) {
                    d[i] /= 16;
                    uint8_t lsb = d[i] & 0x000000FF;
                    buf.emplace_back(lsb);
                }
                stbi_write_png((fileLocation).c_str(), ptr->m_Width, ptr->m_Height, 1, buf.data(),
                               ptr->m_Width);
            }
                break;
            case CRL_COLOR_IMAGE_YUV420:
                // Normalize ycbcr

            {
                Log::Logger::getInstance()->info("Saving Frame: {} from source: {}", ptr->m_Id, stringSrc);

                int width = ptr->m_Width;
                int height = ptr->m_Height;

                AVFrame *src;
                src = av_frame_alloc();
                if (!src) {
                    Log::Logger::getInstance()->error("Could not allocate video frame");
                    break;
                }
                src->format = AV_PIX_FMT_YUV420P;
                src->width = width;
                src->height = height;
                int ret = av_image_alloc(src->data, src->linesize, src->width, src->height,
                                         static_cast<AVPixelFormat>(src->format), 32);
                if (ret < 0) {
                    Log::Logger::getInstance()->error("Could not allocate raw picture buffer");
                    break;
                }

                std::memcpy(src->data[0], ptr->data, ptr->m_Len);
                auto *d = reinterpret_cast<uint16_t *>( ptr->data2);
                for (size_t i = 0; i < ptr->m_Height / 2 * ptr->m_Width / 2; ++i) {
                    src->data[1][i] = d[i] & 0xff;
                    src->data[2][i] = (d[i] >> (8)) & 0xff;
                }

                AVFrame *dst;
                dst = av_frame_alloc();
                if (!dst) {
                    Log::Logger::getInstance()->error("Could not allocate video frame");
                    break;
                }
                dst->format = AV_PIX_FMT_RGB24;
                dst->width = width;
                dst->height = height;
                ret = av_image_alloc(dst->data, dst->linesize, dst->width, dst->height,
                                     static_cast<AVPixelFormat>(dst->format), 8);
                if (ret < 0) {
                    Log::Logger::getInstance()->error("Could not allocate raw picture buffer");
                    break;
                }
                SwsContext *conversion = sws_getContext(width,
                                                        height,
                                                        AV_PIX_FMT_YUV420P,
                                                        width,
                                                        height,
                                                        AV_PIX_FMT_RGB24,
                                                        SWS_FAST_BILINEAR,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr);
                sws_scale(conversion, src->data, src->linesize, 0, height, dst->data, dst->linesize);
                sws_freeContext(conversion);
                stbi_write_png((fileLocation).c_str(), width, height, 3, dst->data[0], dst->linesize[0]);

                av_freep(&src->data[0]);
                av_frame_free(&src);
                av_freep(&dst->data[0]);
                av_frame_free(&dst);

            }

                break;
            case CRL_YUV_PLANAR_FRAME:
                break;
            case CRL_CAMERA_IMAGE_NONE:
                break;
            case CRL_COLOR_IMAGE:
                break;
        }
    }

}
