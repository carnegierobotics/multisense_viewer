//
// Created by magnus on 10/17/23.
//

#ifndef MULTISENSE_VIEWER_RECORDUTILITIES_H
#define MULTISENSE_VIEWER_RECORDUTILITIES_H


#include <filesystem>
#include <tiffio.h>

extern "C" {
#include<libavutil/avutil.h>
#include<libavutil/imgutils.h>
#include <libswscale/swscale.h>
}


#include "Viewer/Scripts/Private/TextureDataDef.h"

namespace RecordUtility {


    static void writeTIFFImage(const std::filesystem::path &fileName, std::shared_ptr<VkRender::TextureData> &ptr,
                               uint32_t bitsPerPixel) {

        int samplesPerPixel = 1;
        TIFF *out = TIFFOpen(fileName.string().c_str(), "w");
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, ptr->m_Width);  // set the width of the image
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, ptr->m_Height);    // set the height of the image
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);   // set number of channels per pixel
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bitsPerPixel);    // set the size of the channels
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

    static void writeTimestamps(const std::filesystem::path &fileName, std::shared_ptr<VkRender::TextureData> &ptr) {
        // Open the CSV file in append mode
        auto timestampFile = fileName.parent_path().parent_path() / "timestamps.csv";
        std::ofstream csvFile(timestampFile, std::ios_base::app);
        // Check if the file opened successfully
        if (!csvFile.is_open()) {
            std::cerr << "Failed to open timestamps.csv for writing." << std::endl;
            return;
        }
        // Write the filename and timestamps to the CSV file
        csvFile << fileName.filename() << "," << ptr->m_TimeSeconds << "." << ptr->m_TimeMicroSeconds << std::endl;
        // Close the file (optional, as the destructor would do this)
        csvFile.close();
    }


    template<typename T>
    static inline std::array<uint8_t, 3> ycbcrToRGB(uint8_t *luma,
                                                    uint8_t *chroma,
                                                    const uint32_t &imageWidth,
                                                    size_t u,
                                                    size_t v) {
        const auto *lumaP = reinterpret_cast<const uint8_t *>(luma);
        const auto *chromaP = reinterpret_cast<const uint8_t *>(chroma);

        const size_t luma_offset = (v * imageWidth) + u;
        const size_t chroma_offset = 2 * (((v / 2) * (imageWidth / 2)) + (u / 2));

        const auto px_y = static_cast<float>(lumaP[luma_offset]);
        const auto px_cb = static_cast<float>(chromaP[chroma_offset + 0]) - 128.0f;
        const auto px_cr = static_cast<float>(chromaP[chroma_offset + 1]) - 128.0f;

        float px_r = px_y + 1.13983f * px_cr;
        float px_g = px_y - 0.39465f * px_cb - 0.58060f * px_cr;
        float px_b = px_y + 2.03211f * px_cb;

        if (px_r < 0.0f) px_r = 0.0f;
        else if (px_r > 255.0f) px_r = 255.0f;
        if (px_g < 0.0f) px_g = 0.0f;
        else if (px_g > 255.0f) px_g = 255.0f;
        if (px_b < 0.0f) px_b = 0.0f;
        else if (px_b > 255.0f) px_b = 255.0f;

        return {{static_cast<uint8_t>(px_r), static_cast<uint8_t>(px_g), static_cast<uint8_t>(px_b)}};
    }

    static inline void
    ycbcrToRGB(uint8_t *luma, uint8_t *chroma, const uint32_t &width,
               const uint32_t &height, uint8_t *output);

    void ycbcrToRGB(uint8_t *luma, uint8_t *chroma, const uint32_t &width,
                    const uint32_t &height, uint8_t *output) {
        const size_t rgb_stride = width * 3;

        for (uint32_t y = 0; y < height; ++y) {
            const size_t row_offset = y * rgb_stride;

            for (uint32_t x = 0; x < width; ++x) {
                memcpy(output + row_offset + (3 * x), ycbcrToRGB<uint8_t>(luma, chroma, width, x, y).data(),
                       3);
            }
        }
    }

    static void writePPMImage(const std::filesystem::path &fileName, std::shared_ptr<VkRender::TextureData> &ptr) {
        std::vector<uint8_t> output(ptr->m_Width * ptr->m_Height * 3);
        ycbcrToRGB(ptr->data, ptr->data2, ptr->m_Width, ptr->m_Height, output.data());
        // something like this

        std::ofstream outputStream(fileName.string(),
                                   std::ios::out | std::ios::binary);
        if (!outputStream.good()) {
            Log::Logger::getInstance()->error("Failed top open file {} for writing", fileName.string());
            return;
        }
        const uint32_t imageSize = ptr->m_Height * ptr->m_Width * 3;
        outputStream << "P6\n"
                     << ptr->m_Width << " " << ptr->m_Height << "\n"
                     << 0xFF << "\n";
        outputStream.write(reinterpret_cast<const char *>(output.data()), imageSize);
        outputStream.close();
    }

    static void writePNGImageGrayscale(const std::filesystem::path &fileName, std::shared_ptr<VkRender::TextureData> &ptr, bool convertDisparity=false){
        auto dataPtr = ptr->data;
        if (convertDisparity){
            auto *d = reinterpret_cast<uint16_t *>( ptr->data);
            std::vector<uint8_t> buf;
            buf.reserve(ptr->m_Width * ptr->m_Height);
            for (size_t i = 0; i < ptr->m_Width * ptr->m_Height; ++i) {
                d[i] /= 16;
                uint8_t lsb = d[i] & 0x000000FF;
                buf.emplace_back(lsb);
            }

            dataPtr = buf.data();
        }

        stbi_write_png((fileName.string()).c_str(), ptr->m_Width, ptr->m_Height, 1, dataPtr,
                       ptr->m_Width);
    }

    static void writePNGImageColor(const std::filesystem::path &fileName, std::shared_ptr<VkRender::TextureData> &ptr){

        int width = ptr->m_Width;
        int height = ptr->m_Height;

        AVFrame *src;
        src = av_frame_alloc();
        if (!src) {
            Log::Logger::getInstance()->error("Could not allocate video frame");
            return;
        }
        src->format = AV_PIX_FMT_YUV420P;
        src->width = width;
        src->height = height;
        int ret = av_image_alloc(src->data, src->linesize, src->width, src->height,
                                 static_cast<AVPixelFormat>(src->format), 32);
        if (ret < 0) {
            Log::Logger::getInstance()->error("Could not allocate raw picture buffer");
            return;
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
            return;
        }
        dst->format = AV_PIX_FMT_RGB24;
        dst->width = width;
        dst->height = height;
        ret = av_image_alloc(dst->data, dst->linesize, dst->width, dst->height,
                             static_cast<AVPixelFormat>(dst->format), 8);
        if (ret < 0) {
            Log::Logger::getInstance()->error("Could not allocate raw picture buffer");
            return;
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
        stbi_write_png((fileName.string()).c_str(), width, height, 3, dst->data[0], dst->linesize[0]);

        av_freep(&src->data[0]);
        av_frame_free(&src);
        av_freep(&dst->data[0]);
        av_frame_free(&dst);

    }
}

#endif //MULTISENSE_VIEWER_RECORDUTILITIES_H
