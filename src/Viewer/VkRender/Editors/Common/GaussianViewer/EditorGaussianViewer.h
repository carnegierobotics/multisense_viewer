//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
#define MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H

#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/VkRender/Scene.h"
#include "Viewer/VkRender/RenderPipelines/GaussianModelGraphicsPipeline.h"

namespace VkRender {

    class EditorGaussianViewer : public Editor {
    public:
        EditorGaussianViewer() = delete;

        explicit EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid = UUID()) : Editor(createInfo, uuid) {
            addUI("EditorUILayer");
            addUI("DebugWindow");
        }

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onUpdate() override;

        ~EditorGaussianViewer() override = default;

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;

        void onSceneLoad() override;

    private:
        Camera m_editorCamera;
        std::shared_ptr<Scene> m_activeScene;
        std::unordered_map<entt::entity, std::unique_ptr<GaussianModelGraphicsPipeline>> m_gaussianRenderPipelines;
    };
}

#endif //MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
