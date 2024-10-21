//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
#define MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H

#include <Viewer/VkRender/Editors/Video/VideoPlaybackSystem.h>

#include "Viewer/VkRender/Editors/PipelineManager.h"
#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"

namespace VkRender {

    class Editor3DViewport : public Editor {
    public:
        Editor3DViewport() = delete;

        explicit Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid);

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


        ~Editor3DViewport() override;

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;

        // TODO all of the component notifications should rather be part of a Renderer class
        // TODO A goal would be to just call Renderer::submit(m_activeScene) which will handle all gpu resources creation and maintenance
        // TODO This will be necessary for SceneRenderers, i.e. rendering the scene with active cameras and use the rendered image as a texture in the scene (Not editorcameras)
        void onComponentAdded(Entity entity, MeshComponent& meshComponent) override;
        void onComponentRemoved(Entity entity, MeshComponent& meshComponent) override;
        void onComponentUpdated(Entity entity, MeshComponent& meshComponent) override;

        void onComponentAdded(Entity entity, MaterialComponent& meshComponent) override;
        void onComponentRemoved(Entity entity, MaterialComponent& meshComponent) override;
        void onComponentUpdated(Entity entity, MaterialComponent& meshComponent) override;

        void onComponentAdded(Entity entity, PointCloudComponent &pointCloudComponent) override;
        void onComponentRemoved(Entity entity, PointCloudComponent &pointCloudComponent) override;
        void onComponentUpdated(Entity entity, PointCloudComponent &pointCloudComponent) override;

        void createDescriptorPool();

        void allocatePerEntityDescriptorSet(uint32_t frameIndex, Entity entity);

        void updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity);

    private:
        Camera m_editorCamera;
        std::reference_wrapper<Camera> m_activeCamera = m_editorCamera;
        std::shared_ptr<Scene> m_activeScene;

        PipelineManager m_pipelineManager;
        std::shared_ptr<VideoPlaybackSystem> m_videoPlaybackSystem;

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

    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
