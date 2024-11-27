//
// Created by mgjer on 14/10/2024.
//

#include "VideoSource.h"

#include <stb_image.h>

namespace VkRender {
    VideoSource::VideoSource(const std::filesystem::path &folderPath, ImageType type, float fps, bool loop)
        : currentFrameIndex(0), timeAccumulator(0.0f), frameDuration(1.0f / fps), imageType(type), loop(loop),
          stopLoading(false) {
        // Load all frame paths from the folder
        for (const auto &entry: std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                framePaths.push_back(entry.path());
            }
        }

        // Sort frame paths to ensure correct order
        std::sort(framePaths.begin(), framePaths.end());

        // Start the loading thread
        startLoadingThread();
    }

    VideoSource::~VideoSource() {
        stopLoadingThread();
    }

    void VideoSource::resetPlayback() {
        // Stop the loading thread temporarily
        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            stopLoading = true;
        }
        bufferCV.notify_all();

        // Wait for the loading thread to acknowledge stopping
        if (loadingThread.joinable()) {
            loadingThread.join();
        }

        // Clear the frame buffer
        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            frameBuffer.clear();
        }

        // Reset indices and accumulators
        currentFrameIndex = 0;
        timeAccumulator = 0.0f;
        stopLoading = false;

        // Restart the loading thread from the beginning
        startLoadingThread();
    }

    void VideoSource::startLoadingThread() {
        loadingThread = std::thread([this]() {
            size_t loadFrameIndex = 0;

            while (!stopLoading) {
                {
                    std::unique_lock<std::mutex> lock(bufferMutex);
                    // Wait if buffer is full
                    bufferCV.wait(lock, [this]() {
                        return frameBuffer.size() < bufferSize || stopLoading;
                    });
                }

                if (stopLoading) {
                    break;
                }

                // Load the next frame
                if (loadFrameIndex >= framePaths.size()) {
                    if (loop) {
                        loadFrameIndex = 0;
                    } else {
                        break; // No more frames to load
                    }
                }

                const auto &framePath = framePaths[loadFrameIndex];
                int width, height, channels;
                void *pixels = nullptr;

                // Load image based on image type
                if (imageType == ImageType::RGB) {
                    stbi_uc *data = stbi_load(framePath.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
                    if (data) {
                        pixels = data;
                        channels = 4; // RGBA
                    }
                } else if (imageType == ImageType::Grayscale) {
                    stbi_uc *data = stbi_load(framePath.string().c_str(), &width, &height, &channels, STBI_grey);
                    if (data) {
                        pixels = data;
                        channels = 1;
                    }
                } else if (imageType == ImageType::Disparity16Bit) {
                    stbi_us *data = stbi_load_16(framePath.string().c_str(), &width, &height, &channels, 1);
                    if (data) {
                        pixels = data;
                        channels = 1; // 16-bit single channel
                    }
                }

                if (pixels) {
                    size_t pixelSize = (imageType == ImageType::Disparity16Bit) ? sizeof(uint16_t) : sizeof(uint8_t);
                    size_t imageSize = width * height * channels * pixelSize;

                    // Copy pixel data into a vector
                    std::vector<uint8_t> pixelData(imageSize);
                    memcpy(pixelData.data(), pixels, imageSize);

                    // Free the original pixel data
                    stbi_image_free(pixels);

                    // Create FrameData
                    FrameData frameData = {std::move(pixelData), width, height, channels, loadFrameIndex};

                    // Add frameData to the buffer
                    {
                        std::lock_guard<std::mutex> lock(bufferMutex);
                        frameBuffer.push_back(std::move(frameData));
                    }
                    bufferCV.notify_one();

                    // Move to the next frame
                    loadFrameIndex++;
                } else {
                    // Handle error (e.g., log warning)
                    // Skip this frame and move to the next
                    loadFrameIndex++;
                }
            }
        });
    }

    void VideoSource::stopLoadingThread() {
        {
            std::lock_guard<std::mutex> lock(bufferMutex);
            stopLoading = true;
        }
        bufferCV.notify_all();
        if (loadingThread.joinable()) {
            loadingThread.join();
        }
    }
} // namespace VkRender
