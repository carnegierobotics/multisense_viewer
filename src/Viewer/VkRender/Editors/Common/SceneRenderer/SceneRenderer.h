//
// Created by mgjer on 30/09/2024.
//

#ifndef MULTISENSE_SCENERENDERER_H
#define MULTISENSE_SCENERENDERER_H

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/RenderPipelines/UboGraphicsPipeline.h"

namespace VkRender {
    class SceneRenderer : public Editor {
    public:
        SceneRenderer() = delete;

        explicit SceneRenderer(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;
        void onRenderDepthOnly(CommandBuffer &drawCmdBuffers) override;

        void onSceneLoad(std::shared_ptr<Scene> scene) override;


        ~SceneRenderer() override {
            if (m_activeScene) {
                m_activeScene->removeDestroyFunction(
                        this); // Unregister the callback since we're destroying the class anyway
                m_activeScene.reset();
            }

        }

    private:
        std::shared_ptr<Camera>  m_activeCamera;
        std::shared_ptr<Scene> m_activeScene;
        std::unordered_map<UUID, std::unique_ptr<GraphicsPipeline>> m_renderPipelines;
        std::unordered_map<UUID, std::unique_ptr<GraphicsPipeline>> m_depthOnlyRenderPipelines;

        void onEntityDestroyed(entt::entity entity);

        void generatePipelines();

        void cleanUpUnusedPipelines();
    };
}


#endif //SCENERENDERER_H
