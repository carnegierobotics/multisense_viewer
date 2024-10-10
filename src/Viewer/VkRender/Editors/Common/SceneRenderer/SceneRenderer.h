//
// Created by mgjer on 30/09/2024.
//

#ifndef MULTISENSE_SCENERENDERER_H
#define MULTISENSE_SCENERENDERER_H

#include <Viewer/VkRender/Editors/PipelineManager.h>

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"

namespace VkRender {
    class SceneRenderer : public Editor {
    public:
        SceneRenderer() = delete;

        explicit SceneRenderer(EditorCreateInfo &createInfo, UUID uuid);

        void onEditorResize() override;

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command);

        void collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> &renderGroups);

        void onComponentAdded(Entity entity, MeshComponent &meshComponent);

        void onComponentRemoved(Entity entity, MeshComponent &meshComponent);

        void onComponentUpdated(Entity entity, MeshComponent &meshComponent);

        void onComponentAdded(Entity entity, MaterialComponent &materialComponent);

        void onComponentRemoved(Entity entity, MaterialComponent &materialComponent);

        void onComponentUpdated(Entity entity, MaterialComponent &materialComponent);

        void createDescriptorPool();

        void allocatePerEntityDescriptorSet(uint32_t frameIndex, Entity entity);

        void updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity);

        std::shared_ptr<MaterialInstance> initializeMaterial(Entity entity, const MaterialComponent &materialComponent);

        std::shared_ptr<MeshInstance> initializeMesh(const MeshComponent &meshComponent);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        ~SceneRenderer() override;

    private:
        std::shared_ptr<Camera>  m_activeCamera;
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


#endif //SCENERENDERER_H
