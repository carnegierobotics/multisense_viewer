//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGE
#define MULTISENSE_VIEWER_EDITORIMAGE

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/VkRender/Core/VulkanTexture.h"

namespace VkRender {
    class GraphicsPipeline2D;


    class EditorImage : public Editor {
    public:
        EditorImage() = delete;

        explicit EditorImage(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        ~EditorImage() override = default;

        void onMouseMove(const MouseButtons &mouse) override;
        void onPipelineReload() override;

        void onFileDrop(const std::filesystem::path &path) override;

        void onMouseScroll(float change) override;

        void onEditorResize() override;

    private:
        std::unique_ptr<GraphicsPipeline2D> m_renderPipelines;

        std::shared_ptr<Scene> m_activeScene;
        std::shared_ptr<Camera> m_activeCamera;

        std::shared_ptr<VulkanTexture2D> m_colorTexture;
        std::shared_ptr<VulkanImage> m_colorImage;

        std::shared_ptr<VulkanTexture2D> m_multiSenseTexture;

        size_t m_playVideoFrameIndex = 0;
    };
}

#endif //MULTISENSE_VIEWER_EDITORIMAGE
