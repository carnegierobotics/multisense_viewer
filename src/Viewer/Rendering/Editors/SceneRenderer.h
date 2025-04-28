//
// Created by mgjer on 30/09/2024.
//

#ifndef MULTISENSE_SCENERENDERER_H
#define MULTISENSE_SCENERENDERER_H

#include "BaseCamera.h"
#include "Viewer/Rendering/Core/PipelineManager.h"
#include "Viewer/Rendering/Core/DescriptorSetManager.h"
#include "Viewer/Rendering/Core/GPUResourceCache.h"
#include "Viewer/Rendering/VulkanMeshResourceManager.h"
#include "Viewer/Rendering/MeshManager.h"

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Core/DescriptorRegistry.h"
#include "Viewer/Rendering/Editors/RenderCommand.h"
#include "Viewer/Rendering/Editors/CpuGpuBuffer.h"

namespace VkRender {
    class SceneRenderer : public Editor {
    public:
        SceneRenderer() = delete;

        explicit SceneRenderer(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void bindResourcesAndDraw(const CommandBuffer &commandBuffer, const RenderCommand &command);

        std::shared_ptr<MeshInstance> initializeMesh(const MeshComponent &meshComponent);

        std::shared_ptr<MaterialInstance> initializeMaterial(Entity entity, const MaterialComponent &materialComponent);

        void debugPrintStats() const;

        std::unordered_map<DescriptorManagerType, VkDescriptorSet> buildMaterialSamplerSet(std::shared_ptr<MaterialInstance> &matInst);

        void collectRenderCommands(
            std::vector<RenderCommand> &renderGroups, uint32_t frameIndex);

        bool isEntityTreeVisible(Entity e) const;

        PipelineKey makeKey(MeshComponent &mc, const MeshInstance &mi, MaterialInstance *mat, Entity &entity);

        PipelineInfo makePipelineInfo(MaterialInstance *mat);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        void onEditorResize() override;

        void updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity);

        void setActiveCamera(const std::shared_ptr<BaseCamera> &cameraPtr) {
            m_activeCamera = cameraPtr;
        }

        std::shared_ptr<BaseCamera> getActiveCamera() const {
            return m_activeCamera.lock();
        }

        ~SceneRenderer() override;

    private:
        std::weak_ptr<BaseCamera> m_activeCamera;
        std::shared_ptr<Scene> m_activeScene;

        PipelineManager m_pipelineManager;
        DescriptorRegistry descriptorRegistry;

        std::unordered_map<UUID, std::shared_ptr<MaterialInstance> > m_materialInstances;

        std::unique_ptr<MeshResourceManager> m_meshResourceManager; // TODO make static and global function
        std::vector<RenderCommand> m_renderGroups;
        std::unordered_map<PipelineKey, InstanceBatch, PipelineKeyHash> m_batches;

        struct EntityRenderData {
            std::vector<std::unique_ptr<Buffer> > cameraBuffer;
            std::vector<std::unique_ptr<Buffer> > modelBuffer;
            std::vector<std::unique_ptr<Buffer> > materialBuffer;
            std::vector<std::unique_ptr<Buffer> > pointCloudBuffer;
        };

        std::unordered_map<UUID, EntityRenderData> m_entityRenderData;

        using CpuUniformBuffer = CpuGpuBuffer<VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT>;
        using CpuStorageBuffer = CpuGpuBuffer<VK_BUFFER_USAGE_STORAGE_BUFFER_BIT>;
        CpuUniformBuffer m_globalUbo;
        CpuStorageBuffer m_transformSsbo;
        CpuStorageBuffer m_materialSsbo;
        std::vector<VkDescriptorSet> m_globalSets;
        std::vector<VkDescriptorSet> m_transformSets;
        std::vector<VkDescriptorSet> m_materialSets;
        std::unordered_map<const MaterialComponent *, uint32_t> m_matIndices;         // lookup from MaterialInstance* → material array index
        std::unordered_map<UUID,uint32_t> m_transformIndexMap;
        std::unordered_map<UUID,uint32_t> m_materialIndexMap;
        std::vector<InstanceTransform>    m_transforms;              // will hold exactly m_transformCursor entries

        uint32_t m_transformCursor = 0;         // per‑frame counters reset each onUpdate
        VkPipelineLayout           m_pipelineLayout; // single layout for all sets

        // helper to (re)create buffers on first scene load or resize
        void createGlobalBuffers();

        // fill the three buffers every frame
        void streamFrameData(uint32_t frameIdx);
        void createGlobalPipelineLayout();

        struct FrameRenderStats {
            /* scene scale */
            uint32_t entityCount = 0; // entities that reached the renderer
            uint32_t batchCount = 0; // InstanceBatch objects
            uint32_t instanceTotal = 0; // sum of all InstanceData written
            uint32_t drawCalls = 0; // vkCmdDraw* actually recorded
            uint32_t drawCallsSaved() const {
                return entityCount > drawCalls
                           ? entityCount - drawCalls
                           : 0;
            }

            /* timings (nanoseconds) */
            uint64_t cpuCollectNs = 0; // collectRenderCommands()
            uint64_t cpuRecordNs = 0; // bindResourcesAndDraw() loop
            uint64_t gpuNs = 0; // GPU timestamp interval
            uint64_t presentNs = 0; // swap-chain present (optional)

            /* helpers ---------------------------------------------------------- */
            void reset() { *this = {}; }
            double asMs(uint64_t ns) const { return ns * 1e-6; }
        };

        struct CpuTimer {
            using clk = std::chrono::high_resolution_clock;
            clk::time_point t0;
            void start() { t0 = clk::now(); }

            uint64_t ns() const {
                return std::chrono::duration_cast<std::chrono::nanoseconds>
                        (clk::now() - t0).count();
            }
        };

        FrameRenderStats m_stats{};
        VkQueryPool m_timestampPool{};

    public:
        void fillPipelineKey(PipelineKey &key, MeshComponent &meshC, MeshInstance &meshInst, Entity entity);

        std::shared_ptr<MaterialInstance> getMaterialInstance(Entity entity);

        std::unordered_map<DescriptorManagerType, VkDescriptorSet> buildCommonDescriptorSets(
            Entity entity, uint32_t frameIdx, std::shared_ptr<MaterialInstance> mat);
    };
}


#endif //SCENERENDERER_H
