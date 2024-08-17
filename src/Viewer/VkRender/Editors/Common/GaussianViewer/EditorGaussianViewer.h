//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
#define MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H

#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Core/Camera.h"
#include "Viewer/VkRender/Scene.h"
#include "Viewer/VkRender/RenderPipelines/3DGS/GaussianModelGraphicsPipeline.h"
#include "Viewer/VkRender/RenderPipelines/GraphicsPipeline2D.h"

namespace VkRender {

    class EditorGaussianViewer : public Editor {
    public:
        EditorGaussianViewer() = delete;

        explicit EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid = UUID()) : Editor(createInfo, uuid) {
            addUI("EditorUILayer");
            addUI("DebugWindow");
            addUI("EditorGaussianViewerLayer");
        }

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onUpdate() override;

        ~EditorGaussianViewer() override = default;

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;

        void onSceneLoad() override;

        void onKeyCallback(const Input& input) override;

    private:
        Camera* m_activeCamera;
        std::shared_ptr<Scene> m_activeScene;
        std::unordered_map<entt::entity, std::unique_ptr<GaussianModelGraphicsPipeline>> m_gaussianRenderPipelines;
        std::unordered_map<entt::entity, std::unique_ptr<GraphicsPipeline2D>> m_2DRenderPipeline;

        void generatePipelines();
    };
}

#endif //MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
