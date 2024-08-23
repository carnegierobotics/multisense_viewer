//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORIMAGE
#define MULTISENSE_VIEWER_EDITORIMAGE

#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Scene.h"
#include "Viewer/VkRender/RenderPipelines/GraphicsPipeline.h"

namespace VkRender {

    class EditorImage : public Editor {
    public:
        EditorImage() = delete;

        explicit EditorImage(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        ~EditorImage() override = default;

        void onMouseMove(const MouseButtons &mouse) override;

        void onFileDrop(const std::filesystem::path& path) override;

        void onMouseScroll(float change) override;
        void onEditorResize() override;

    private:
       std::unique_ptr<GraphicsPipeline> m_renderPipelines;
       entt::registry m_registry;
       bool m_recreateOnNextImageChange = false;
       std::unique_ptr<ImageComponent> imageComponent;

    };
}

#endif //MULTISENSE_VIEWER_EDITORIMAGE
