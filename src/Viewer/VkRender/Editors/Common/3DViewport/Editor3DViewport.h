//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
#define MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H

#include "Viewer/VkRender/Editor.h"
#include "Viewer/Scenes/Default/DefaultScene.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/RenderPipelines/CameraModelGraphicsPipeline.h"

namespace VkRender {

    class Editor3DViewport : public Editor {
    public:
        Editor3DViewport() = delete;

        explicit Editor3DViewport(EditorCreateInfo &createInfo);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onSceneLoad() override;


        ~Editor3DViewport() override {
            if (m_activeScene) {
                m_activeScene->removeDestroyFunction(
                        this); // Unregister the callback since we're destroying the class anyway
                m_activeScene.reset();
            }

        }

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;

    private:
        Camera m_editorCamera;
        std::shared_ptr<Scene> m_activeScene;
        std::unordered_map<entt::entity, std::unique_ptr<DefaultGraphicsPipeline>> m_renderPipelines;
        std::unordered_map<entt::entity, std::unique_ptr<CameraModelGraphicsPipeline>> m_cameraRenderPipelines;

        void onEntityDestroyed(entt::entity entity);
    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
