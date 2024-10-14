//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Video/VideoPlaybackSystem.h"
#include "Viewer/Application/Application.h"


namespace VkRender {
    // VideoSource.cpp
    VideoSource::VideoSource(const std::filesystem::path &folderPath, ImageType type, float fps, bool loop)
        : currentFrameIndex(0), timeAccumulator(0.0f), frameDuration(1.0f / fps), imageType(type), loop(loop) {
        // Load all frame paths from the folder
        for (const auto &entry: std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                framePaths.push_back(entry.path());
            }
        }

        // Sort frame paths to ensure correct order
        std::sort(framePaths.begin(), framePaths.end());

        // Initialize the texture based on image type
        // We'll create an empty texture here; it will be filled during the first update
        // The texture format will depend on the imageType
    }

    size_t VideoPlaybackSystem::addVideoSource(
        const std::filesystem::path &folderPath,
        VideoSource::ImageType type,
        float fps,
        bool loop) {
        VideoSource videoSource(folderPath, type, fps, loop);

        // Create an initial texture based on image type
        VkFormat format;
        switch (type) {
            case VideoSource::ImageType::RGB:
                format = VK_FORMAT_R8G8B8A8_UNORM;
                break;
            case VideoSource::ImageType::Grayscale:
                format = VK_FORMAT_R8_UNORM;
                break;
            case VideoSource::ImageType::Disparity16Bit:
                format = VK_FORMAT_R16_UNORM;
                break;
            default:
                format = VK_FORMAT_R8G8B8A8_UNORM;
                break;
        }

        // Create an empty texture with the appropriate format
        videoSource.texture = createEmptyTexture(format);

        m_videoSources.push_back(std::move(videoSource));
        return m_videoSources.size() - 1; // Return the index of the added video source
    }

    void VideoPlaybackSystem::update(float deltaTime) {
        for (auto &video: m_videoSources) {
            video.timeAccumulator += deltaTime;

            // Check if it's time to advance to the next frame
            while (video.timeAccumulator >= video.frameDuration) {
                video.timeAccumulator -= video.frameDuration;
                video.currentFrameIndex++;

                if (video.currentFrameIndex >= video.framePaths.size()) {
                    if (video.loop) {
                        video.currentFrameIndex = 0;
                    } else {
                        video.currentFrameIndex = video.framePaths.size() - 1; // Stay on the last frame
                    }
                }

                // Load the next frame and update the texture
                const auto &framePath = video.framePaths[video.currentFrameIndex];
                int width, height, channels;
                void *pixels = nullptr;

                // Load image based on the image type
                if (video.imageType == VideoSource::ImageType::RGB) {
                    stbi_uc *data = stbi_load(framePath.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
                    pixels = data;
                    channels = 4; // RGBA
                } else if (video.imageType == VideoSource::ImageType::Grayscale) {
                    stbi_uc *data = stbi_load(framePath.string().c_str(), &width, &height, &channels, STBI_grey);
                    pixels = data;
                    channels = 1;
                } else if (video.imageType == VideoSource::ImageType::Disparity16Bit) {
                    stbi_us *data = stbi_load_16(framePath.string().c_str(), &width, &height, &channels, 1);
                    pixels = data;
                    channels = 1; // 16-bit single channel
                }

                if (pixels) {
                    VkDeviceSize imageSize = width * height * channels * (
                                                 (video.imageType == VideoSource::ImageType::Disparity16Bit)
                                                     ? sizeof(uint16_t)
                                                     : sizeof(uint8_t));

                    // Update the texture with the new frame data
                    video.texture->loadImage(pixels, imageSize);

                    // Free the image data
                    if (video.imageType == VideoSource::ImageType::Disparity16Bit) {
                        stbi_image_free(pixels);
                    } else {
                        stbi_image_free(pixels);
                    }
                } else {
                    // Handle error (e.g., log warning)
                }
            }
        }
    }

    std::shared_ptr<VulkanTexture2D> VideoPlaybackSystem::getTexture(size_t index) const {
        if (index < m_videoSources.size()) {
            return m_videoSources[index].texture;
        }
        return nullptr;
    }

    std::shared_ptr<VulkanTexture2D> VideoPlaybackSystem::createEmptyTexture(VkFormat format) {
        // Create an empty VulkanTexture2D with the specified format
        // You may need to specify the initial dimensions; these can be updated when the first frame is loaded
        VulkanImageCreateInfo imageCreateInfo;
        // Set imageCreateInfo parameters based on format
        // ...

        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = std::make_shared<VulkanImage>(imageCreateInfo);

        return std::make_shared<VulkanTexture2D>(textureCreateInfo);
    }
}
