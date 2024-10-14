//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_VideoPlaybackSystem
#define MULTISENSE_VIEWER_VideoPlaybackSystem

#include "VideoSource.h"
#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {

    // VideoPlaybackSystem.h
    class VideoPlaybackSystem {
    public:
        explicit VideoPlaybackSystem(Application *context);

        size_t addVideoSource(const std::filesystem::path &folderPath, VideoSource::ImageType type, float fps, bool loop);
        void update(float deltaTime);
        std::shared_ptr<VulkanTexture2D> getTexture(size_t index) const;

        void resetAllSourcesPlayback();

    private:
        Application *m_context;
        std::vector<std::unique_ptr<VideoSource>> m_videoSources;

        std::shared_ptr<VulkanTexture2D> createEmptyTexture(VkFormat format, uint32_t width, uint32_t height) const;
    };
;
}


#endif //MULTISENSE_VIEWER_VideoPlaybackSystem
