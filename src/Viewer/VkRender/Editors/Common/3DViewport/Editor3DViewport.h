//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
#define MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H

#include <Viewer/VkRender/Editors/DescriptorSetManager.h>
#include <Viewer/VkRender/Editors/Video/VideoPlaybackSystem.h>
#include <Viewer/VkRender/RenderResources/GraphicsPipeline2D.h>

#include "Viewer/VkRender/Editors/PipelineManager.h"
#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Editors/Common/SceneRenderer.h"

namespace VkRender {

    class Editor3DViewport : public Editor {
    public:
        Editor3DViewport() = delete;

        explicit Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;


        void onSceneLoad(std::shared_ptr<Scene> scene) override;
        void onEditorResize() override;

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;
        void onKeyCallback(const Input &input) override;

        std::shared_ptr<MeshInstance> setupMesh();

        void onRenderSettingsChanged();

        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand &command);

        void collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups, uint32_t
            frameIndex);

    private:
        std::shared_ptr<Camera> m_editorCamera;
        CameraComponent* m_lastActiveCamera = nullptr;
        std::shared_ptr<Scene> m_activeScene;

        SceneRenderer* m_sceneRenderer;
        std::shared_ptr<VulkanTexture2D> m_colorTexture;

        std::vector<std::unique_ptr<Buffer>> m_shaderSelectionBuffer;
        // Quad and descriptor setup
        PipelineManager m_pipelineManager;
        DescriptorRegistry m_descriptorRegistry;
        std::shared_ptr<MeshInstance> m_meshInstances;

    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
