//
// Created by mgjer on 14/10/2024.
//

#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H

#include <thread>
#include <deque>

#include "Viewer/Application/pch.h"
#include "Viewer/VkRender/Core/VulkanTexture.h"

namespace VkRender {

    class VideoSource {
    public:
        enum class ImageType {
            RGB,
            Grayscale,
            Disparity16Bit
        };

        VideoSource(const std::filesystem::path &folderPath, ImageType type, float fps, bool loop);
        ~VideoSource();

        void resetPlayback();

        // Delete copy constructor and copy assignment operator
        VideoSource(const VideoSource&) = delete;
        VideoSource& operator=(const VideoSource&) = delete;

        // Delete move constructor and move assignment operator
        VideoSource(VideoSource&&) = delete;
        VideoSource& operator=(VideoSource&&) = delete;

        // Existing members
        std::vector<std::filesystem::path> framePaths;
        size_t currentFrameIndex;
        float timeAccumulator;
        float frameDuration; // Time per frame (e.g., 1/30 for 30 FPS)
        ImageType imageType;
        std::shared_ptr<VulkanTexture2D> texture;
        bool loop;

        // New members for buffering and threading
        std::thread loadingThread;
        std::atomic<bool> stopLoading;
        std::mutex bufferMutex;
        std::condition_variable bufferCV;

        struct FrameData {
            std::vector<uint8_t> pixels; // Pixel data
            int width;
            int height;
            int channels;
            size_t frameIndex;
        };

        std::deque<FrameData> frameBuffer; // Buffer of preloaded frames
        size_t bufferSize = 15; // Desired buffer size

        void startLoadingThread();
        void stopLoadingThread();
    };

} // namespace VkRender
#endif //VIDEOSOURCE_H
