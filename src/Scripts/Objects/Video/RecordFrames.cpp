/**
 * @file: MultiSense-Viewer/src/Scripts/Objects/Video/RecordFrames.cpp
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/
#include "Viewer/Scripts/Objects/Video/RecordFrames.h"

#include <filesystem>
#include <tiffio.h>

extern "C" {
#include<libavutil/avutil.h>
#include<libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

void RecordFrames::setup() {
    threadPool = std::make_unique<VkRender::ThreadPool>(3);
}

void RecordFrames::update() {
    if (saveImage) {
        // For each enabled source in all windows
        for (auto &src: sources) {
            if (src == "Idle")
                continue;

            // Check if we saved a lot more images of this type than others. So skip it
            bool skipThisFrame = false;
            auto count = savedImageSourceCount.find(src);
            std::string otherSrc;
            for (const auto &i: savedImageSourceCount) {
                if (src == i.first)
                    continue;
                if (count->second > i.second + 3) {
                    skipThisFrame = true;
                    otherSrc = i.first;
                    break;
                }
            }
            if (skipThisFrame) {
                Log::Logger::getInstance()->trace(
                        "Skipping saving frame {} of source {} since we have to many compared to {} which has {}",
                        savedImageSourceCount[src] + 1, src, otherSrc, savedImageSourceCount[otherSrc]);
                continue;
            }

            // For each remote head index
            for (crl::multisense::RemoteHeadChannel remoteIdx = crl::multisense::Remote_Head_0;
                 remoteIdx <=
                 (isRemoteHead ? crl::multisense::Remote_Head_3 : crl::multisense::Remote_Head_0); ++remoteIdx) {
                const auto &conf = renderData.crlCamera->getCameraInfo(remoteIdx).imgConf;
                auto tex = std::make_shared<VkRender::TextureData>(Utils::CRLSourceToTextureType(src), conf.width(),
                                                                   conf.height(), false, true);

                if (renderData.crlCamera->getCameraStream(src, tex.get(), remoteIdx)) {
                    if (src == "Color Aux" || src == "Color Rectified Aux") {
                        if (tex->m_Id != tex->m_Id2)
                            continue;
                    }
                    if (threadPool->getTaskListSize() < MAX_IMAGES_IN_QUEUE) {
                        threadPool->Push(saveImageToFile, Utils::CRLSourceToTextureType(src), saveImagePath, src,
                                         remoteIdx, tex, isRemoteHead, compression);
                        savedImageSourceCount[src]++;
                    } else if (threadPool->getTaskListSize() >= MAX_IMAGES_IN_QUEUE && compression == "tiff") {
                        Log::Logger::getInstance()->info("Record image queue is full. Starting to drop frames");

                    }
                }
            }
        }
    }
    if (savePointCloud) {

        const auto &conf = renderData.crlCamera->getCameraInfo(0).imgConf;
        auto disparityTex = std::make_shared<VkRender::TextureData>(CRL_DISPARITY_IMAGE, conf.width(), conf.height(),
                                                                    false, true);

        std::shared_ptr<VkRender::TextureData> colorTex;
        CRLCameraDataType texType = CRL_GRAYSCALE_IMAGE;
        std::string source = "Luma Rectified Left";
        if (useAuxColor) {
            source = "Color Rectified Aux";
            texType = CRL_COLOR_IMAGE_YUV420;
        }
        colorTex = std::make_shared<VkRender::TextureData>(texType, conf.width(), conf.height(), false, true);

        // If we successfully fetch both image streams into a texture
        if (renderData.crlCamera->getCameraStream(source, colorTex.get(), 0) &&
            renderData.crlCamera->getCameraStream("Disparity Left", disparityTex.get(), 0)) {
            // Calculate pointcloud then save to file
            if (threadPool->getTaskListSize() < MAX_IMAGES_IN_QUEUE) {
                threadPool->Push(savePointCloudToPlyFile, saveImagePathPointCloud, disparityTex,
                                 colorTex, useAuxColor,
                                 renderData.crlCamera->getCameraInfo(0).QMat,
                                 renderData.crlCamera->getCameraInfo(0).pointCloudScale,
                                 renderData.crlCamera->getCameraInfo(0).focalLength
                );
            }
        }
    }
}

// Function to calculate the hash of a vector of strings
size_t RecordFrames::hashVector(const std::vector<std::string> &v) {
    size_t seed = 0;
    for (const auto &str: v) {
        std::hash<std::string> hasher;
        seed ^= hasher(str) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}


void RecordFrames::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    for (VkRender::Device &dev: uiHandle->devices) {


        if (dev.state != CRL_STATE_ACTIVE)
            continue;

        sources.clear();
        for (const auto &window: dev.win) {
            if (!Utils::isInVector(sources, window.second.selectedSource)) {
                sources.emplace_back(window.second.selectedSource);

                // Check if "foo" exists in the map
                if (savedImageSourceCount.find(window.second.selectedSource) == savedImageSourceCount.end() &&
                    window.second.selectedSource != "Idle") {
                    Log::Logger::getInstance()->trace("Added source {} to saved image counter in record frames",
                                                      window.second.selectedSource);
                    savedImageSourceCount[window.second.selectedSource] = 0;
                }
            }

        }

        if (hashVector(sources) != hashVector(prevSources)){
            savedImageSourceCount.clear();

        }

        prevSources = sources;

        saveImagePath = dev.outputSaveFolder;
        saveImagePathPointCloud = dev.outputSaveFolderPointCloud;
        saveImage = dev.isRecording;
        compression = dev.saveImageCompressionMethod;
        isRemoteHead = dev.isRemoteHead;
        savePointCloud = dev.isRecordingPointCloud;
        useAuxColor = dev.useAuxForPointCloudColor == 1;
    }
}

void RecordFrames::savePointCloudToPlyFile(const std::filesystem::path &saveDirectory,
                                           std::shared_ptr<VkRender::TextureData> &depthTex,
                                           std::shared_ptr<VkRender::TextureData> &colorTex, bool useAuxColor,
                                           const glm::mat4 &Q, const float &scale, const float &focalLength) {

    std::filesystem::path fileLocation = saveDirectory;
    fileLocation.append(std::to_string(depthTex->m_Id) + ".ply");
    // Dont overwrite already saved images
    if (std::filesystem::exists(fileLocation))
        return;
    // Create folders if no exists
    if (!std::filesystem::is_directory(saveDirectory) ||
        !std::filesystem::exists(saveDirectory)) { // Check if src folder exists
        std::filesystem::create_directories(saveDirectory); // create src folder
    }

    struct WorldPoint {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float r = 0.0f, g = 0.0f, b = 0.0f;
    };


    const auto *disparityP = reinterpret_cast<const uint16_t *>(depthTex->data);
    const auto *leftRectifiedP = reinterpret_cast<const uint8_t *>(colorTex->data);
    uint32_t width = depthTex->m_Width;
    uint32_t height = depthTex->m_Height;
    uint32_t minDisparity = 5;

    std::vector<WorldPoint> points;
    points.reserve(height * width);

    std::vector<uint8_t> colorRGB(colorTex->m_Width * colorTex->m_Height * 3);
    if (useAuxColor) {
        RecordFrames::ycbcrToRGB(colorTex->data, colorTex->data2, colorTex->m_Width, colorTex->m_Height,
                                 colorRGB.data());
    }

    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {

            const size_t index = h * width + w;

            //
            // MultiSense 16 bit disparity images are stored in 1/16 of a pixel. This allows us to send subpixel
            // resolutions with integer values

            const float d = static_cast<float>(disparityP[index]) / 16.0f;

            if (d < minDisparity) {
                continue;
            }
            float invB = focalLength / d;
            glm::vec4 imgCoords(w, h, d, 1);
            glm::vec4 worldCoords = Q * imgCoords * (1 / invB);
            worldCoords = worldCoords / worldCoords.w * scale;

            if (useAuxColor) {
                const size_t colorIndex = (h * width + w) * 3;
                points.emplace_back(WorldPoint{worldCoords.x,
                                               worldCoords.y,
                                               worldCoords.z,
                                               static_cast<float>(colorRGB[colorIndex]),
                                               static_cast<float>(colorRGB[colorIndex + 1]),
                                               static_cast<float>(colorRGB[colorIndex + 2])});

            } else {
                points.emplace_back(WorldPoint{worldCoords.x,
                                               worldCoords.y,
                                               worldCoords.z,
                                               static_cast<float>(leftRectifiedP[index]),
                                               static_cast<float>(leftRectifiedP[index]),
                                               static_cast<float>(leftRectifiedP[index])});
            }
        }
    }

    std::stringstream ss;

    ss << "ply\n";
    ss << "format ascii 1.0\n";
    ss << "element vertex " << points.size() << "\n";
    ss << "property float x\n";
    ss << "property float y\n";
    ss << "property float z\n";
    ss << "property uchar red\n";
    ss << "property uchar green\n";
    ss << "property uchar blue\n";
    ss << "end_header\n";

    for (const auto &point: points) {
        ss << point.x << " " << point.y << " " << point.z << " " << point.r << " " << point.g << " " << point.b << "\n";
    }

    std::ofstream ply(fileLocation.c_str());
    ply << ss.str();
}

void RecordFrames::saveImageToFile(CRLCameraDataType type, const std::string &path, std::string &stringSrc,
                                   crl::multisense::RemoteHeadChannel remoteHead,
                                   std::shared_ptr<VkRender::TextureData> &ptr, bool isRemoteHead,
                                   std::string &compression) {
    // Create folders. One for each Source
    std::filesystem::path saveDirectory{};
    std::replace(stringSrc.begin(), stringSrc.end(), ' ', '_');
    if (type == CRL_COLOR_IMAGE_YUV420 && compression != "png") {
        compression = "ppm";
    }

    if (isRemoteHead) {
        saveDirectory = path + "/Head_" + std::to_string(remoteHead + 1) + "/" + stringSrc + "/" + compression + "/";
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
    if (compression == "tiff" || compression == "ppm") {
        switch (type) {
            case CRL_POINT_CLOUD:
                break;
            case CRL_GRAYSCALE_IMAGE: {
                int samplesPerPixel = 1;
                TIFF *out = TIFFOpen(fileLocation.string().c_str(), "w");
                TIFFSetField(out, TIFFTAG_IMAGEWIDTH, ptr->m_Width);  // set the width of the image
                TIFFSetField(out, TIFFTAG_IMAGELENGTH, ptr->m_Height);    // set the height of the image
                TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);   // set number of channels per pixel
                TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    // set the size of the channels
                TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
                //   Some other essential fields to set that you do not have to understand for now.
                TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
                TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);

                // We set the strip size of the file to be size of one row of pixels
                TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1);
                //Now writing image to the file one strip at a time
                for (uint32_t row = 0; row < ptr->m_Height; row++) {
                    if (TIFFWriteScanline(out, &(ptr->data[row * ptr->m_Width]), row, 0) < 0)
                        break;
                }
                TIFFClose(out);
            }
                break;
            case CRL_DISPARITY_IMAGE: {
                int samplesPerPixel = 1;
                TIFF *out = TIFFOpen(fileLocation.string().c_str(), "w");
                TIFFSetField(out, TIFFTAG_IMAGEWIDTH, ptr->m_Width);  // set the width of the image
                TIFFSetField(out, TIFFTAG_IMAGELENGTH, ptr->m_Height);    // set the height of the image
                TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);   // set number of channels per pixel
                TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);    // set the size of the channels
                TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
                //   Some other essential fields to set that you do not have to understand for now.
                TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
                // We set the strip size of the file to be size of one row of pixels
                TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, 1);
                //Now writing image to the file one strip at a time
                for (uint32_t row = 0; row < ptr->m_Height; row++) {
                    if (TIFFWriteScanline(out, &(ptr->data[row * ptr->m_Width * sizeof(uint16_t)]), row, 0) < 0)
                        break;
                }
                TIFFClose(out);
            }
                break;
            case CRL_COLOR_IMAGE_YUV420:
                // Normalize ycbcr
            {
                std::vector<uint8_t> output(ptr->m_Width * ptr->m_Height * 3);
                RecordFrames::ycbcrToRGB(ptr->data, ptr->data2, ptr->m_Width, ptr->m_Height, output.data());
                // something like this

                std::ofstream outputStream((fileLocation.string()).c_str(),
                                           std::ios::out | std::ios::binary);
                if (!outputStream.good()) {
                    Log::Logger::getInstance()->error("Failed top open file {} for writing", fileLocation.string());
                    break;
                }
                const uint32_t imageSize = ptr->m_Height * ptr->m_Width * 3;
                outputStream << "P6\n"
                             << ptr->m_Width << " " << ptr->m_Height << "\n"
                             << 0xFF << "\n";
                outputStream.write(reinterpret_cast<const char *>(output.data()), imageSize);
                outputStream.close();
                break;
            }
            case CRL_CAMERA_IMAGE_NONE:
            case CRL_COLOR_IMAGE_RGBA:
                break;
            default:
                Log::Logger::getInstance()->info("Recording not specified for this format");
                break;
        }
    } else if (compression == "png") {
        switch (type) {
            case CRL_POINT_CLOUD:
                break;
            case CRL_GRAYSCALE_IMAGE: {

                stbi_write_png((fileLocation.string()).c_str(), ptr->m_Width, ptr->m_Height, 1, ptr->data,
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
                stbi_write_png((fileLocation.string()).c_str(), ptr->m_Width, ptr->m_Height, 1, buf.data(),
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
                stbi_write_png((fileLocation.string()).c_str(), width, height, 3, dst->data[0], dst->linesize[0]);

                av_freep(&src->data[0]);
                av_frame_free(&src);
                av_freep(&dst->data[0]);
                av_frame_free(&dst);

            }

                break;
            default:
                Log::Logger::getInstance()->info("Recording not specified for this format");
                break;
        }
    }

}


void RecordFrames::ycbcrToRGB(uint8_t *luma, uint8_t *chroma, const uint32_t &width,
                              const uint32_t &height, uint8_t *output) {
    const size_t rgb_stride = width * 3;

    for (uint32_t y = 0; y < height; ++y) {
        const size_t row_offset = y * rgb_stride;

        for (uint32_t x = 0; x < width; ++x) {
            memcpy(output + row_offset + (3 * x), RecordFrames::ycbcrToRGB<uint8_t>(luma, chroma, width, x, y).data(),
                   3);
        }
    }
}