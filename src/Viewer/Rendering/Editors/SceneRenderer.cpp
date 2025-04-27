//
// Created by mgjer on 30/09/2024.
//


#include "Viewer/Rendering/Editors/SceneRenderer.h"

#include <Viewer/Rendering/Components/LightSourceComponent.h>
#include <Viewer/Rendering/Core/VulkanShaderModule.h>

#include "Viewer/Rendering/Components/Components.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"

#include "Viewer/Application/Application.h"

#include "Viewer/Scenes/Entity.h"

#include "Viewer/Rendering/MeshInstance.h"
#include "Viewer/Assets/ShaderLoader.h"


namespace VkRender {
    void MeshInstance::ensureInstanceBuffer(VkDeviceSize size, VulkanDevice &dev) {
        if (instanceBuffer && instanceBuffer->m_size >= size) return;

        dev.createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, // must include VERTEX_BUFFER_BIT!
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // simplest for frequently-updated data
            instanceBuffer,
            size,
            nullptr);
    }

    SceneRenderer::SceneRenderer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        m_renderToOffscreen = true;
        m_activeCamera = std::make_shared<BaseCamera>(static_cast<float>(m_createInfo.width) / m_createInfo.height);
        descriptorRegistry.createManager(DescriptorManagerType::MVP, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::Material, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorManagerType::DynamicCameraGizmo, m_context->vkDevice());

        m_meshResourceManager = std::make_unique<MeshResourceManager>(m_context);
        /*
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = 2;
        vkCreateQueryPool(m_context->vkDevice().m_LogicalDevice, &qp, nullptr, &m_timestampPool);
        */
    }

    void SceneRenderer::onEditorResize() {
    }

    SceneRenderer::~SceneRenderer() {
        m_entityRenderData.clear();
    }

    void SceneRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
        // Once we load a scene we need to create pipelines according to the objects specified in the scene.
        // For OBJModels we are alright with a default rendering pipeline (Phong lightining and stuff)
        // The pipelines also define memory handles between CPU and GPU. It makes more logical scenes if these attributes belong to the OBJModelComponent
        // But we need it accessed in the pipeline
        m_activeScene = m_context->activeScene();
        // TODO not clear what this does but it creates render reosurces of the editor was copied as part of a split operation
        auto view = m_activeScene->getRegistry().view<IDComponent>();
        for (auto e: view) {
            auto entity = Entity(e, m_activeScene.get());
            auto name = entity.getName();
            if (entity.hasComponent<MaterialComponent>()) {
                onComponentAdded(entity, entity.getComponent<MaterialComponent>());
            }
            if (entity.hasComponent<PointCloudComponent>()) {
                onComponentAdded(entity, entity.getComponent<PointCloudComponent>());
            }
            if (entity.hasComponent<MeshComponent>()) {
                onComponentAdded(entity, entity.getComponent<MeshComponent>());
            }
        }
    }

    void SceneRenderer::onUpdate() {
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        auto view = m_activeScene->getRegistry().view<IDComponent>();

        std::vector<Entity> lightEntities;
        for (auto e: view) {
            auto entity = Entity(e, m_activeScene.get());
            if (entity.hasComponent<LightSourceComponent>()) {
                lightEntities.push_back(entity);
            }
        }

        uint32_t frameIndex = m_context->currentFrameIndex();
        for (auto e: view) {
            auto entity = Entity(e, m_activeScene.get());

            if (entity.hasComponent<MeshComponent>()) {
                GlobalUniformBufferObject globalUBO = {};
                auto activeCameraPtr = m_activeCamera.lock(); // Lock to get shared_ptr
                if (activeCameraPtr) {
                    globalUBO.view = activeCameraPtr->matrices.view;
                    globalUBO.projection = activeCameraPtr->matrices.projection;
                    globalUBO.cameraPosition = activeCameraPtr->matrices.position;
                }

                // Map and copy data to the global uniform buffer
                void *data;
                vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                            m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory, 0,
                            sizeof(globalUBO),
                            0,
                            &data);
                memcpy(data, &globalUBO, sizeof(globalUBO));
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                              m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory);
            }
            if (entity.hasComponent<TransformComponent>() && entity.hasComponent<MeshComponent>()) {
                void *data;
                auto &transformComponent = m_activeScene->getRegistry().get<TransformComponent>(entity);
                vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                            m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex]->m_memory, 0, VK_WHOLE_SIZE, 0,
                            &data);
                auto *modelMatrices = reinterpret_cast<glm::mat4 *>(data);
                *modelMatrices = transformComponent.getTransform();
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                              m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex]->m_memory);
            }
            if (entity.hasComponent<MaterialComponent>() && !m_entityRenderData[entity.getUUID()].materialBuffer.
                empty()) {
                auto &material = entity.getComponent<MaterialComponent>();
                MaterialBufferObject matUBO = {};
                matUBO.baseColor = material.albedo;
                matUBO.specular = material.specular;
                matUBO.diffuse = material.diffuse;
                matUBO.emissiveFactor = glm::vec4(material.emission);
                matUBO.useVertexColor = material.useTexture;


                for (int i = 0; i < lightEntities.size(); ++i) {
                    matUBO.lightPosition[i] = glm::vec4(
                        lightEntities[i].getComponent<TransformComponent>().getPosition(), 1.0f);
                }
                matUBO.numLightSources = static_cast<float>(lightEntities.size());
                assert(matUBO.numLightSources < 10);

                void *data;
                vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                            m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory, 0,
                            sizeof(MaterialBufferObject), 0,
                            &data);
                memcpy(data, &matUBO, sizeof(MaterialBufferObject));
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                              m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory);
            }
        }
    }


    void SceneRenderer::onRender(CommandBuffer &commandBuffer) {
        m_stats.reset(); // start fresh
        CpuTimer tFrame;
        tFrame.start();

        CpuTimer tCollect;
        tCollect.start();
        collectRenderCommands(m_renderGroups, commandBuffer.getActiveFrameIndex());
        m_stats.cpuCollectNs = tCollect.ns();

        /* ---------- 2. record timestamp before first draw --------------- */
        //VkCommandBuffer vkCB = commandBuffer.getActiveBuffer();
        //vkCmdResetQueryPool(vkCB, m_timestampPool, 0, 2);
        //vkCmdWriteTimestamp(vkCB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_timestampPool, 0); // start

        /* ---------- 3. record draw calls -------------------------------- */
        //CpuTimer tRecord;
        //tRecord.start();
        for (auto &rc: m_renderGroups)
            bindResourcesAndDraw(commandBuffer, rc);
        //m_stats.cpuRecordNs = tRecord.ns();

        /* ---------- 4. GPU timestamp after last draw -------------------- */
        //vkCmdWriteTimestamp(vkCB, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timestampPool, 1); // end


        /* ---------- 5. gather GPU duration after submission ------------- */
        // call once you know the cmdBuf has finished (frame fence)

        /*
        uint64_t timestamps[2];
        if (vkGetQueryPoolResults(m_context->vkDevice().m_LogicalDevice,
                                  m_timestampPool,
                                  0, 2,
                                  sizeof(timestamps), timestamps,
                                  sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        {
            uint64_t period = m_context->vkDevice().m_deviceProps.limits.timestampPeriod; // ns
            m_stats.gpuNs = (timestamps[1] - timestamps[0]) * period;
        }
        */

        /* ---------- 6. total draw-call info ----------------------------- */
        m_stats.drawCalls = static_cast<uint32_t>(m_renderGroups.size());

        debugPrintStats();
    }

    void SceneRenderer::bindResourcesAndDraw(const CommandBuffer &commandBuffer,
                                             const RenderCommand &cmd) {
        VkCommandBuffer cb = commandBuffer.getActiveBuffer();

        /* --- pipeline & descriptor sets ----------------------------------- */
        vkCmdBindPipeline(cb,
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          cmd.pipeline->getPipeline());

        for (auto &[setIndex, setHandle]: cmd.descriptorSets) {
            if (setHandle == VK_NULL_HANDLE) continue; // skip gaps
            vkCmdBindDescriptorSets(cb,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    cmd.pipeline->getPipelineLayout(),
                                    static_cast<uint32_t>(setIndex), 1, &setHandle,
                                    0, nullptr);
        }

        /* --- vertex & instance buffers ------------------------------------ */
        VkBuffer vbos[2];
        VkDeviceSize offs[2]{0, 0};
        uint32_t vbCount = 0;

        /* binding 0 : per-vertex */
        vbos[vbCount++] = cmd.meshInstance->vertexBuffer->m_buffer;

        /* binding 1 : per-instance (only if it exists) */
        if (cmd.meshInstance->instanceBuffer &&
            cmd.meshInstance->instanceCount > 0)
            vbos[vbCount++] = cmd.meshInstance->instanceBuffer->m_buffer;

        vkCmdBindVertexBuffers(cb, 0, vbCount, vbos, offs);

        /* optional index buffer */
        if (cmd.meshInstance->indexBuffer)
            vkCmdBindIndexBuffer(cb,
                                 cmd.meshInstance->indexBuffer->m_buffer,
                                 0, VK_INDEX_TYPE_UINT32);

        /* --- draw ---------------------------------------------------------- */
        const uint32_t ic = std::max(cmd.meshInstance->instanceCount, 1u);

        if (cmd.meshInstance->indexBuffer) // indexed draw
            vkCmdDrawIndexed(cb,
                             cmd.meshInstance->indexCount,
                             ic,
                             0, 0, 0);
        else // non-indexed draw
            vkCmdDraw(cb,
                      cmd.meshInstance->vertexCount,
                      ic,
                      0, 0);
    }

    void SceneRenderer::collectRenderCommands(
        std::vector<RenderCommand> &renderGroups, uint32_t frameIndex) {
        m_batches.clear(); // m_batches : unordered_map<PipelineKey, InstanceBatch>
        auto group = m_activeScene->getRegistry().group<MeshComponent, TransformComponent>();

        for (auto [e, meshComponent, transformComponent]: group.each()) {
            m_stats.entityCount++; // every visible entity

            Entity entity(e, m_activeScene.get());
            if (!isEntityTreeVisible(entity)) // ← your old visibility test wrapped
                continue;
            std::string tag = entity.getName();
            UUID uuid = entity.getUUID();

            //std::unordered_map<DescriptorManagerType, VkDescriptorSet> descriptorSets; // Add the descriptor set here
            //std::unordered_map<DescriptorManagerType, std::vector<VkWriteDescriptorSet> > descriptorWritesTracker;

            /* ----------------------------------------------------------------- */
            /* pipeline-key & resources that are the same for every instance     */
            /* ----------------------------------------------------------------- */
            auto meshData = MeshManager::instance().getMeshData(meshComponent);
            if (!meshData) continue;

            auto meshInst = m_meshResourceManager->getMeshInstance(
                meshComponent.getCacheIdentifier(),
                meshData,
                meshComponent.meshDataType());
            if (!meshInst) continue;

            std::shared_ptr<MaterialInstance> matInst = getMaterialInstance(entity);


            auto descriptors = buildCommonDescriptorSets(entity,
                                                         frameIndex,
                                                         matInst);

            PipelineKey key = makeKey(meshComponent, *meshInst, matInst.get());


            /* -------- find-or-create the batch --------------------------- */
            auto &batch = m_batches[key];
            if (!batch.mesh) // first entity with this key
            {
                batch.mesh = meshInst;
                batch.material = matInst;
                batch.sets = descriptors;
                batch.key = key;
            }

            /* -------- push per-instance payload -------------------------- */
            batch.cpuInstances.push_back({transformComponent.getTransform()});
        }

        /* 2.  turn every batch into one RenderCommand ------------------------ */
        m_renderGroups.clear();
        RenderPassInfo rp{};
        rp.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        rp.renderPass = m_renderPass->getRenderPass();
        rp.debugName = "SceneRenderer::";

        m_stats.batchCount = static_cast<uint32_t>(m_batches.size());

        for (auto &[key, batch]: m_batches) {
            /* --- upload cpuInstances → mesh->instanceBuffer ----------------- */
            const VkDeviceSize bytes = batch.cpuInstances.size() * sizeof(InstanceData);
            batch.mesh->ensureInstanceBuffer(bytes, m_context->vkDevice());

            void *dst = nullptr;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        batch.mesh->instanceBuffer->m_memory,
                        0, bytes, 0, &dst);
            memcpy(dst, batch.cpuInstances.data(), bytes);
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          batch.mesh->instanceBuffer->m_memory);

            batch.mesh->instanceCount = static_cast<uint32_t>(batch.cpuInstances.size());
            PipelineInfo pipelineInfo = makePipelineInfo(batch.material.get());

            /* --- build the single draw command ------------------------------ */
            RenderCommand cmd;
            cmd.pipeline = m_pipelineManager.getOrCreatePipeline(key, pipelineInfo, rp, m_context);
            cmd.meshInstance = batch.mesh.get();
            cmd.materialInstance = batch.material.get();
            cmd.descriptorSets = std::move(batch.sets);

            m_renderGroups.emplace_back(std::move(cmd));
            m_stats.instanceTotal += batch.mesh->instanceCount;
        }
    }

    bool SceneRenderer::isEntityTreeVisible(Entity e) const {
        // Initialize a flag to determine if we should skip this entity
        bool skipEntity = false;

        // Start with the current entity
        Entity current = e;
        // Traverse up the parent hierarchy
        while (current.getParent()) {
            // Move to the parent entity
            current = current.getParent();
            // Check if the parent has both GroupComponent and VisibilityComponent
            if (current.hasComponent<GroupComponent>() && current.hasComponent<VisibleComponent>()) {
                // Retrieve the VisibilityComponent
                auto &visibility = current.getComponent<VisibleComponent>();
                // If visibility is set to false, mark to skip this entity
                if (!visibility.visible) {
                    skipEntity = true;
                    break; // No need to check further ancestors
                }
            }
        }
        // If an ancestor with visible == false was found, skip to the next entity
        if (skipEntity) {
            return false;
        }
        return !skipEntity;
    }

    PipelineKey SceneRenderer::makeKey(MeshComponent &mc,
                                       const MeshInstance &mi,
                                       MaterialInstance *mat) {
        PipelineKey k{};

        /* fixed-function  */
        k.topology = mi.topology;
        k.polygonMode = mc.polygonMode();

        /* shared ids  */
        k.meshId = Utils::crc32(mc.getCacheIdentifier());
        k.vsCRC = Utils::crc32("BlinnPhongShaderInstanced.vert");
        k.fsCRC = Utils::crc32("BlinnPhongShaderInstanced.frag");
        k.materialFlags = mat && mat->baseColorTexture ? 1u : 0u;
        return k;
    }

    PipelineInfo SceneRenderer::makePipelineInfo(MaterialInstance *mat) {

        PipelineInfo info{};
        /* descriptor set layouts */
        info.setLayouts.resize(mat ? 2 : 1);

        info.setLayouts[0] = descriptorRegistry.getManager(DescriptorManagerType::MVP)
                .getDescriptorSetLayout();
        info.setLayouts[1] = mat
                              ? descriptorRegistry.getManager(DescriptorManagerType::Material)
                              .getDescriptorSetLayout()
                              : VK_NULL_HANDLE;

        /* vertex input: binding 0 (mesh), binding 1 (instance) */
        info.bindings = {
            {
                {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX},
                {1, sizeof(InstanceData), VK_VERTEX_INPUT_RATE_INSTANCE}
            }
        };

        info.attrs = {
            {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 6},
                {3, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 8},
                {4, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 10},
                {5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
                {6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
                {7, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32},
                {8, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48}
            }
        };
        info.attrCount = 9;

        info.materialInstance = mat;

        return info;
    }

    std::shared_ptr<MaterialInstance>
    SceneRenderer::getMaterialInstance(Entity entity) {
        if (!entity.hasComponent<MaterialComponent>()) return nullptr;

        auto &matC = entity.getComponent<MaterialComponent>();
        auto it = m_materialInstances.find(entity.getUUID());
        if (it != m_materialInstances.end()) return it->second;

        auto mi = initializeMaterial(entity, matC);
        m_materialInstances[entity.getUUID()] = mi;
        return mi;
    }

    std::unordered_map<DescriptorManagerType, VkDescriptorSet>
    SceneRenderer::buildCommonDescriptorSets(Entity entity,
                                             uint32_t frameIdx,
                                             std::shared_ptr<MaterialInstance> mat) {
        std::unordered_map<DescriptorManagerType, VkDescriptorSet> out;


        /* MVP set (camera only now) ---------------------------------------- */
        auto &rd = m_entityRenderData[entity.getUUID()];
        VkWriteDescriptorSet camWrite{};
        camWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        camWrite.dstBinding = 0;
        camWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        camWrite.descriptorCount = 1;
        camWrite.pBufferInfo = &rd.cameraBuffer[frameIdx]->m_descriptorBufferInfo;

        out[DescriptorManagerType::MVP] =
                descriptorRegistry.getManager(DescriptorManagerType::MVP)
                .getOrCreateDescriptorSet({camWrite});


        /* material set (unchanged) ---------------------------------------- */
        if (mat) {
            std::vector<VkWriteDescriptorSet> writes(2);
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &rd.materialBuffer[frameIdx]->m_descriptorBufferInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].descriptorCount = 1;
            writes[1].pImageInfo = &mat->baseColorTexture->getDescriptorInfo();

            out[DescriptorManagerType::Material] =
                    descriptorRegistry.getManager(DescriptorManagerType::Material)
                    .getOrCreateDescriptorSet(writes);
        }
        return out;
    }


    void SceneRenderer::onComponentAdded(Entity entity, MeshComponent &meshComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        m_entityRenderData[entity.getUUID()].cameraBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].modelBuffer.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int frameIndex = 0; frameIndex < m_context->swapChainBuffers().size(); ++frameIndex) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex],
                sizeof(GlobalUniformBufferObject), nullptr, "SceneRenderer:MeshComponent:Camera",
                m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex],
                sizeof(glm::mat4), nullptr, "SceneRenderer:MeshComponent:Model",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MeshComponent &meshComponent) {
        if (m_entityRenderData.contains(entity.getUUID())) {
            m_entityRenderData[entity.getUUID()].cameraBuffer.clear();
            m_entityRenderData[entity.getUUID()].modelBuffer.clear();
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MeshComponent &meshComponent) {
    }

    void SceneRenderer::onComponentAdded(Entity entity, MaterialComponent &materialComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        if (m_materialInstances.contains(entity.getUUID())) {
            m_materialInstances.erase(entity.getUUID());
        }
        m_entityRenderData[entity.getUUID()].materialBuffer.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int i = 0; i < m_context->swapChainBuffers().size(); ++i) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].materialBuffer[i],
                sizeof(MaterialBufferObject), nullptr, "SceneRenderer:MaterialComponent",
                m_context->getDebugUtilsObjectNameFunction());;
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MaterialComponent &materialComponent) {
        if (m_materialInstances.contains(entity.getUUID())) {
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MaterialComponent &materialComponent) {
        // add a video source if selected
        if (m_materialInstances.contains(
            entity.getUUID())) {
            // TODO look into just replacing what changed instead of erasing, triggering a new pipeline creation. However, the cost for recreating everything in a material is very small
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, PointCloudComponent &pointCloudComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        m_entityRenderData[entity.getUUID()].cameraBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].modelBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].pointCloudBuffer.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int i = 0; i < m_context->swapChainBuffers().size(); ++i) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].cameraBuffer[i],
                sizeof(GlobalUniformBufferObject), nullptr, "SceneRenderer:PointCloudComponent:Camera",
                m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].modelBuffer[i],
                sizeof(glm::mat4), nullptr, "SceneRenderer:PointCloudComponent:Model",
                m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].pointCloudBuffer[i],
                sizeof(PointCloudUBO), nullptr, "SceneRenderer:PointCloudComponent:PC",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, PointCloudComponent &pointCloudComponent) {
    }

    void SceneRenderer::onComponentUpdated(Entity entity, PointCloudComponent &pointCloudComponent) {
    }

    void SceneRenderer::updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity) {
        // Get the active camera entity
        // Compute view and projection matrices
    }


    std::shared_ptr<MaterialInstance> SceneRenderer::initializeMaterial(
        Entity entity, const MaterialComponent &materialComponent) {
        auto materialInstance = std::make_shared<MaterialInstance>();
        if (std::filesystem::exists(materialComponent.albedoTexturePath)) {
            materialInstance->baseColorTexture = EditorUtils::createTextureFromFile(materialComponent.albedoTexturePath,
                m_context);
        } else {
            materialInstance->baseColorTexture = EditorUtils::createEmptyTexture(300, 300, VK_FORMAT_R8G8B8A8_UNORM,
                m_context, VMA_MEMORY_USAGE_GPU_ONLY, true);
        }
        // 1 Load Shader code
        auto vsSPV = assetManager()->get<SPIRVAsset>(materialComponent.vertexShaderName.string());
        auto fsSPV = assetManager()->get<SPIRVAsset>(materialComponent.fragmentShaderName.string());
        // 2) Wrap into a GPU resource
        VulkanShaderModuleCreateInfo vertexShaderCreateInfo(m_context->vkDevice(), vsSPV, VK_SHADER_STAGE_VERTEX_BIT, materialComponent.vertexShaderName.string());
        VulkanShaderModuleCreateInfo vertexShaderCreateInfo2(m_context->vkDevice(), vsSPV, VK_SHADER_STAGE_VERTEX_BIT, materialComponent.vertexShaderName.string());
        VulkanShaderModuleCreateInfo fragmentShaderCreateInfo(m_context->vkDevice(), fsSPV, VK_SHADER_STAGE_FRAGMENT_BIT, materialComponent.fragmentShaderName.string());
        // 3) Later in pipeline creation:

        // 3) Ask the GPU cache for shared modules:
        auto vsModule = cache()->shaderModules.get(vertexShaderCreateInfo);
        auto fsModule = cache()->shaderModules.get(fragmentShaderCreateInfo);

        materialInstance->addShader(vertexShaderCreateInfo);
        materialInstance->addShader(fragmentShaderCreateInfo);

        Log::Logger::getInstance()->info("Created Material for Entity: {}", entity.getName());
        return materialInstance;
    }

    void SceneRenderer::debugPrintStats() const {
        Log::Logger::getInstance()->trace(
            "Frame: ent={}  batches={}  instances={}  draws={}  saved={}"
            " | CPU collect {:.2f} ms  record {:.2f} ms | GPU {:.2f} ms",
            m_stats.entityCount,
            m_stats.batchCount,
            m_stats.instanceTotal,
            m_stats.drawCalls,
            m_stats.drawCallsSaved(),
            m_stats.asMs(m_stats.cpuCollectNs),
            m_stats.asMs(m_stats.cpuRecordNs),
            m_stats.asMs(m_stats.gpuNs));
    }
}
