//
// Created by mgjer on 30/09/2024.
//

#ifndef MULTISENSE_SCENERENDERER_H
#define MULTISENSE_SCENERENDERER_H

#include <multisense_viewer/src/Viewer/VkRender/Editors/PipelineManager.h>

#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Editors/Video/VideoPlaybackSystem.h"

namespace VkRender {
    class SceneRenderer : public Editor {
    public:
        SceneRenderer() = delete;

        explicit SceneRenderer(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand &command);

        std::shared_ptr<MeshInstance> initializeMesh(const MeshComponent &meshComponent);

        std::shared_ptr<MaterialInstance> initializeMaterial(Entity entity, MaterialComponent & materialComponent);

        void updateMaterialDescriptors(Entity entity, MaterialInstance *materialInstance);

        std::shared_ptr<PointCloudInstance> initializePointCloud(Entity entity, PointCloudComponent &pointCloudComponent);

        void updatePointCloudDescriptors(Entity entity, PointCloudInstance *pointCloudInstance);

        void collectRenderCommands(
                std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> &renderGroups);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;
        void onEditorResize() override;

        void createDescriptorPool();

        void allocatePerEntityDescriptorSet(uint32_t frameIndex, Entity entity);

        void updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity);

        ~SceneRenderer() override;

        Camera& getCamera(){
            return m_activeCamera;
        }

    private:
        Camera m_activeCamera;
        std::shared_ptr<Scene> m_activeScene;

        PipelineManager m_pipelineManager;
        //std::shared_ptr<VideoPlaybackSystem> m_videoPlaybackSystem;

        std::unordered_map<UUID, std::shared_ptr<MaterialInstance>> m_materialInstances;
        std::unordered_map<UUID, std::shared_ptr<MeshInstance>> m_meshInstances;
        std::unordered_map<UUID, std::shared_ptr<PointCloudInstance>> m_pointCloudInstances;

        struct EntityRenderData {
            // Mesh rendering and typical shader buffers
            std::vector<std::unique_ptr<Buffer>> cameraBuffer;
            std::vector<std::unique_ptr<Buffer>> modelBuffer;
            std::vector<VkDescriptorSet> descriptorSets;

            // Material stuff
            std::vector<std::unique_ptr<Buffer>> materialBuffer;
            std::vector<VkDescriptorSet> materialDescriptorSets;
            // Pount cloud rendering
            std::vector<std::unique_ptr<Buffer>> pointCloudBuffer;
            std::vector<VkDescriptorSet> pointCloudDescriptorSets;
        };
        std::unordered_map<UUID, EntityRenderData> m_entityRenderData;

        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_materialDescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_pointCloudDescriptorSetLayout = VK_NULL_HANDLE;

    private:

        void onComponentAdded(Entity entity, MeshComponent& meshComponent) override;
        void onComponentRemoved(Entity entity, MeshComponent& meshComponent) override;
        void onComponentUpdated(Entity entity, MeshComponent& meshComponent) override;

        void onComponentAdded(Entity entity, MaterialComponent& meshComponent) override;
        void onComponentRemoved(Entity entity, MaterialComponent& meshComponent) override;
        void onComponentUpdated(Entity entity, MaterialComponent& meshComponent) override;

        void onComponentAdded(Entity entity, PointCloudComponent &pointCloudComponent) override;
        void onComponentRemoved(Entity entity, PointCloudComponent &pointCloudComponent) override;
        void onComponentUpdated(Entity entity, PointCloudComponent &pointCloudComponent) override;
    };
}


#endif //SCENERENDERER_H
