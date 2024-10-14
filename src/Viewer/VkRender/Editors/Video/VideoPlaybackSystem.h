//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_VideoPlaybackSystem
#define MULTISENSE_VIEWER_VideoPlaybackSystem

#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {
    struct VideoSource {
        enum class ImageType {
            RGB,
            Grayscale,
            Disparity16Bit
        };

        std::vector<std::filesystem::path> framePaths;
        size_t currentFrameIndex;
        float timeAccumulator;
        float frameDuration; // Time per frame (e.g., 1/30 for 30 FPS)
        ImageType imageType;
        std::shared_ptr<VulkanTexture2D> texture;
        bool loop;

        VideoSource(const std::filesystem::path& folderPath, ImageType type, float fps, bool loop);
    };

    // VideoPlaybackSystem.h
    class VideoPlaybackSystem {
    public:
        explicit VideoPlaybackSystem(Application* context);

        void update(float deltaTime);
        size_t addVideoSource(const std::filesystem::path& folderPath, VideoSource::ImageType type, float fps, bool loop);
        void removeVideoSource(size_t index);
        std::shared_ptr<VulkanTexture2D> getTexture(size_t index) const;

        std::shared_ptr<VulkanTexture2D> createEmptyTexture(VkFormat format);

    private:
        std::vector<VideoSource> m_videoSources;
        Application* m_context;
    };
;
}


#endif //MULTISENSE_VIEWER_VideoPlaybackSystem
