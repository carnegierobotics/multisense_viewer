//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
#define MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H

#include "Viewer/VkRender/Editors/PipelineManager.h"
#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"

namespace VkRender {

    class Editor3DViewport : public Editor {
    public:
        Editor3DViewport() = delete;

        explicit Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand &command);

        std::shared_ptr<MeshInstance> initializeMesh(const MeshComponent &meshComponent);

        std::shared_ptr<MaterialInstance> initializeMaterial(Entity entity, const MaterialComponent & materialComponent);

        void collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> &renderGroups);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;
        void onEditorResize() override;


        ~Editor3DViewport() override;

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;

        void onComponentAdded(Entity entity, MeshComponent& meshComponent) override;
        void onComponentRemoved(Entity entity, MeshComponent& meshComponent) override;
        void onComponentUpdated(Entity entity, MeshComponent& meshComponent) override;

        void onComponentAdded(Entity entity, MaterialComponent& meshComponent) override;
        void onComponentRemoved(Entity entity, MaterialComponent& meshComponent) override;
        void onComponentUpdated(Entity entity, MaterialComponent& meshComponent) override;

        void createDescriptorPool();

        void allocatePerEntityDescriptorSet(uint32_t frameIndex, Entity entity);

        void updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity);

    private:
        Camera m_editorCamera;
        std::reference_wrapper<Camera> m_activeCamera = m_editorCamera;
        std::shared_ptr<Scene> m_activeScene;

        PipelineManager m_pipelineManager;
        std::unordered_map<UUID, std::shared_ptr<MaterialInstance>> m_materialInstances;
        std::unordered_map<UUID, std::shared_ptr<MeshInstance>> m_meshInstances;

        struct EntityRenderData {
            std::vector<Buffer> cameraBuffer;
            std::vector<Buffer> modelBuffer;
            std::vector<VkDescriptorSet> descriptorSets;

            std::vector<Buffer> materialBuffer;
            std::vector<VkDescriptorSet> materialDescriptorSets;
        };
        std::unordered_map<UUID, EntityRenderData> m_entityRenderData;

        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_materialDescriptorSetLayout = VK_NULL_HANDLE;

    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
