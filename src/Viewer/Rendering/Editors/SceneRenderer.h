//
// Created by mgjer on 30/09/2024.
//

#ifndef MULTISENSE_SCENERENDERER_H
#define MULTISENSE_SCENERENDERER_H

#include "Viewer/Rendering/Core/PipelineManager.h"
#include "Viewer/Rendering/Core/DescriptorSetManager.h"
#include "Viewer/Rendering/VulkanMeshResourceManager.h"
#include "Viewer/Rendering/MeshManager.h"

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Core/DescriptorRegistry.h"
#include "Viewer/Rendering/Editors/RenderCommand.h"

namespace VkRender {
    class SceneRenderer : public Editor {
    public:
        SceneRenderer() = delete;

        explicit SceneRenderer(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand &command);

        std::shared_ptr<MeshInstance> initializeMesh(const MeshComponent &meshComponent);

        std::shared_ptr<MaterialInstance> initializeMaterial(Entity entity, const MaterialComponent & materialComponent);

        void collectRenderCommands(
            std::vector<RenderCommand>& renderGroups, uint32_t frameIndex);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;
        void onEditorResize() override;

        void updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity);

        void setActiveCamera(const std::shared_ptr<Camera>& cameraPtr){
            m_activeCamera = cameraPtr;
        }
        std::shared_ptr<Camera> getActiveCamera() const {
            return m_activeCamera.lock();
        }
        ~SceneRenderer() override;



    private:
        std::weak_ptr<Camera> m_activeCamera;
        std::shared_ptr<Scene> m_activeScene;

        PipelineManager m_pipelineManager;
        DescriptorRegistry descriptorRegistry;

        std::unordered_map<UUID, std::shared_ptr<MaterialInstance>> m_materialInstances;

        std::unique_ptr<MeshResourceManager> m_meshResourceManager;
        MeshManager m_meshManager;
        std::vector<RenderCommand> m_renderGroups;

        struct EntityRenderData {
            std::vector<std::unique_ptr<Buffer>> cameraBuffer;
            std::vector<std::unique_ptr<Buffer>> modelBuffer;
            std::vector<std::unique_ptr<Buffer>> materialBuffer;
            std::vector<std::unique_ptr<Buffer>> pointCloudBuffer;
        };
        std::unordered_map<UUID, EntityRenderData> m_entityRenderData;
    public:

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
