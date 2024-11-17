//
// Created by mgjer on 30/09/2024.
//


#include "SceneRenderer.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "CommonEditorFunctions.h"


namespace VkRender {
    SceneRenderer::SceneRenderer(EditorCreateInfo& createInfo, UUID uuid) : Editor(createInfo, uuid) {
        m_renderToOffscreen = true;
        m_activeCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);
        descriptorRegistry.createManager(DescriptorType::MVP, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorType::Material, m_context->vkDevice());
        descriptorRegistry.createManager(DescriptorType::DynamicCameraGizmo, m_context->vkDevice());
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
        for (auto e : view) {
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
        for (auto entity : view) {
            updateGlobalUniformBuffer(m_context->currentFrameIndex(), Entity(entity, m_activeScene.get()));
        }
    }


    void SceneRenderer::onRender(CommandBuffer& commandBuffer) {
        std::vector<RenderCommand> renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.getActiveFrameIndex());

        // Render each group
        for (auto& command : renderGroups) {
            bindResourcesAndDraw(commandBuffer, command);
        }
    }

    void SceneRenderer::bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command) {
        // Bind vertex buffers
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, command.pipeline->pipeline()->getPipeline());
        for (auto& [index, descriptorSet] : command.descriptorSets) {
            vkCmdBindDescriptorSets(
                cmdBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                command.pipeline->pipeline()->getPipelineLayout(),
                index,
                1,
                &descriptorSet,
                0,
                nullptr
            );
        }

        if (command.meshInstance->vertexBuffer) {
            VkBuffer vertexBuffers[] = {command.meshInstance->vertexBuffer->m_buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);
            // Bind index buffer if the mesh has indices
            if (command.meshInstance->indexBuffer) {
                vkCmdBindIndexBuffer(cmdBuffer, command.meshInstance->indexBuffer->m_buffer, 0,
                                     VK_INDEX_TYPE_UINT32);
            }
        }

        if (command.meshInstance->vertexBuffer) {
            // Issue the draw call
            if (command.meshInstance->indexBuffer) {
                // Indexed draw call
                vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
            }
            else {
                // Non-indexed draw call
                vkCmdDraw(cmdBuffer, command.meshInstance->vertexCount, 1, 0, 0);
            }
        }
        if (command.meshInstance->m_type == CAMERA_GIZMO) {
            vkCmdDraw(cmdBuffer, command.meshInstance->vertexCount, 1, 0, 0);
        }
    }

    void SceneRenderer::collectRenderCommands(
        std::vector<RenderCommand>& renderGroups, uint32_t frameIndex) {
        auto view = m_activeScene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto e : view) {
            Entity entity(e, m_activeScene.get());
            std::string tag = entity.getName();
            UUID uuid = entity.getUUID();
            auto& meshComponent = entity.getComponent<MeshComponent>();
            std::unordered_map<uint32_t, VkDescriptorSet> descriptorSets; // Add the descriptor set here
            PipelineKey key = {};
            key.setLayouts.resize(3);

            std::vector<VkWriteDescriptorSet> descriptorWrites(2);
            auto& renderData = m_entityRenderData[entity.getUUID()];

            {
                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &renderData.cameraBuffer[frameIndex]->m_descriptorBufferInfo;
                descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].pBufferInfo = &renderData.modelBuffer[frameIndex]->m_descriptorBufferInfo;
                VkDescriptorSet mvpDescriptorSet = descriptorRegistry.getManager(DescriptorType::MVP).
                                                                      getOrCreateDescriptorSet(descriptorWrites);
                descriptorSets[0] = mvpDescriptorSet;
                key.setLayouts[0] = descriptorRegistry.getManager(DescriptorType::MVP).getDescriptorSetLayout();
            }
            // Use default descriptor set layout


            // Create the pipeline key based on whether the material exists or not
            // Ensure the MeshInstance exists for this MeshComponent
            std::shared_ptr<MeshInstance> meshInstance;
            auto it = m_meshInstances.find(uuid);
            if (it == m_meshInstances.end()) {
                // Initialize the MeshInstance
                meshInstance = initializeMesh(meshComponent);
                if (!meshInstance)
                    continue;
                Log::Logger::getInstance()->info("Created Mesh for Entity: {}", entity.getName());
                m_meshInstances[uuid] = meshInstance;
            }
            else {
                meshInstance = it->second;
            }
            key.topology = meshInstance->topology;
            key.polygonMode = meshComponent.polygonMode;
            // Check if the entity has a MaterialComponent
            std::shared_ptr<MaterialInstance> materialInstance = nullptr;
            if (entity.hasComponent<MaterialComponent>()) {
                auto& materialComponent = entity.getComponent<MaterialComponent>();
                auto materialIt = m_materialInstances.find(uuid);
                if (materialIt == m_materialInstances.end()) {
                    materialInstance = initializeMaterial(entity, materialComponent);
                    m_materialInstances[uuid] = materialInstance;
                }
                else {
                    materialInstance = materialIt->second;
                }
                key.vertexShaderName = materialComponent.vertexShaderName;
                key.fragmentShaderName = materialComponent.fragmentShaderName;
                key.renderMode = materialInstance->renderMode;
                key.materialPtr = reinterpret_cast<uint64_t*>(materialInstance.get());

                descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[0].dstBinding = 0;
                descriptorWrites[0].dstArrayElement = 0;
                descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[0].descriptorCount = 1;
                descriptorWrites[0].pBufferInfo = &renderData.materialBuffer[frameIndex]->m_descriptorBufferInfo;
                descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[1].dstBinding = 1;
                descriptorWrites[1].dstArrayElement = 0;
                descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrites[1].descriptorCount = 1;
                descriptorWrites[1].pImageInfo = &materialInstance->baseColorTexture->getDescriptorInfo();
                VkDescriptorSet materialDescriptorSet = descriptorRegistry.getManager(DescriptorType::Material).getOrCreateDescriptorSet(descriptorWrites);
                descriptorSets[1] = materialDescriptorSet;
            }
            key.setLayouts[1] = descriptorRegistry.getManager(DescriptorType::Material).getDescriptorSetLayout();


            std::vector<VkVertexInputBindingDescription> vertexInputBinding = {
                {0, sizeof(VkRender::Vertex), VK_VERTEX_INPUT_RATE_VERTEX}
            };
            std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 6},
                {3, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 8},
                {4, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 10},
            };
            key.vertexInputBindingDescriptions = vertexInputBinding;
            key.vertexInputAttributes = vertexInputAttributes;

            if (meshComponent.m_type == CAMERA_GIZMO) {
                key.vertexShaderName = "CameraGizmo.vert";
                key.fragmentShaderName = "default.frag";
                key.vertexInputBindingDescriptions.clear();
                key.vertexInputAttributes.clear();
                std::vector<VkWriteDescriptorSet> descriptorWrite(1);
                descriptorWrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrite[0].dstBinding = 0;
                descriptorWrite[0].dstArrayElement = 0;
                descriptorWrite[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrite[0].descriptorCount = 1;
                descriptorWrite[0].pBufferInfo = &renderData.uboVertexBuffer[frameIndex]->m_descriptorBufferInfo;
                VkDescriptorSet dynamicCameraDescriptorSet = descriptorRegistry.getManager(
                    DescriptorType::DynamicCameraGizmo).getOrCreateDescriptorSet(descriptorWrite);
                descriptorSets[2] = dynamicCameraDescriptorSet;
                // Use default descriptor set layout
            }
            key.setLayouts[2] = descriptorRegistry.getManager(DescriptorType::DynamicCameraGizmo).getDescriptorSetLayout();
            // Create or retrieve the pipeline
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            renderPassInfo.debugName = "SceneRenderer::";

            auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
            // Create the render command

            RenderCommand command;
            command.descriptorSets = descriptorSets;
            command.entity = entity;
            command.pipeline = pipeline;
            command.meshInstance = meshInstance.get();
            command.materialInstance = materialInstance.get(); // May be null if no material is attached
            // Add to render group
            renderGroups.push_back(command);
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, MeshComponent& meshComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        if (m_meshInstances.contains(entity.getUUID())) {
            m_meshInstances.erase(entity.getUUID());
        }
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
        auto& renderData = m_entityRenderData[entity.getUUID()];
        renderData.uboVertexBuffer.resize(m_context->swapChainBuffers().size());
        for (int frameIndex = 0; frameIndex < m_context->swapChainBuffers().size(); ++frameIndex) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_entityRenderData[entity.getUUID()].uboVertexBuffer[frameIndex],
                sizeof(glm::vec4) * 21, nullptr, "SceneRenderer:MeshComponent:uboVertexBuffer",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MeshComponent& meshComponent) {
        if (m_meshInstances.contains(entity.getUUID())) {
            m_meshInstances.erase(entity.getUUID());
        }
        if (m_entityRenderData.contains(entity.getUUID())) {
            m_entityRenderData[entity.getUUID()].cameraBuffer.clear();
            m_entityRenderData[entity.getUUID()].modelBuffer.clear();
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MeshComponent& meshComponent) {
        if (m_meshInstances.contains(entity.getUUID())) {
            m_meshInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, MaterialComponent& materialComponent) {
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

    void SceneRenderer::onComponentRemoved(Entity entity, MaterialComponent& materialComponent) {
        if (m_materialInstances.contains(entity.getUUID())) {
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MaterialComponent& materialComponent) {
        // add a video source if selected
        if (m_materialInstances.contains(
            entity.getUUID())) {
            // TODO look into just replacing what changed instead of erasing, triggering a new pipeline creation. However, the cost for recreating everything in a material is very small
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, PointCloudComponent& pointCloudComponent) {
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

    void SceneRenderer::onComponentRemoved(Entity entity, PointCloudComponent& pointCloudComponent) {
    }

    void SceneRenderer::onComponentUpdated(Entity entity, PointCloudComponent& pointCloudComponent) {
    }

    void SceneRenderer::updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity) {
        // Get the active camera entity
        // Compute view and projection matrices
        if (entity.hasComponent<MeshComponent>()) {
            GlobalUniformBufferObject globalUBO = {};
            auto activeCameraPtr = m_activeCamera.lock(); // Lock to get shared_ptr
            if (activeCameraPtr) {
                globalUBO.view = activeCameraPtr->matrices.view;
                globalUBO.projection = activeCameraPtr->matrices.perspective;
                globalUBO.cameraPosition = activeCameraPtr->pose.pos;
            }

            // Map and copy data to the global uniform buffer
            void* data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory, 0, sizeof(globalUBO),
                        0,
                        &data);
            memcpy(data, &globalUBO, sizeof(globalUBO));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory);

            if (entity.getComponent<MeshComponent>().m_type == CAMERA_GIZMO) {
                void* data;
                vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                            m_entityRenderData[entity.getUUID()].uboVertexBuffer[frameIndex]->m_memory, 0,
                            sizeof(glm::vec4) * 21, 0, &data);
                MeshData meshData(CAMERA_GIZMO, "");
                memcpy(data, meshData.cameraGizmoVertices.data(), sizeof(glm::vec4) * 21);
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                              m_entityRenderData[entity.getUUID()].uboVertexBuffer[frameIndex]->m_memory);
            }
        }
        if (entity.hasComponent<TransformComponent>() && entity.hasComponent<MeshComponent>()) {
            void* data;
            auto& transformComponent = m_activeScene->getRegistry().get<TransformComponent>(entity);
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex]->m_memory, 0, VK_WHOLE_SIZE, 0,
                        &data);
            auto* modelMatrices = reinterpret_cast<glm::mat4*>(data);
            *modelMatrices = transformComponent.getTransform();
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex]->m_memory);
        }
        if (entity.hasComponent<MaterialComponent>() && !m_entityRenderData[entity.getUUID()].materialBuffer.empty()) {
            auto& material = entity.getComponent<MaterialComponent>();
            MaterialBufferObject matUBO = {};
            matUBO.baseColor = material.baseColor;
            matUBO.metallic = material.metallic;
            matUBO.roughness = material.roughness;
            matUBO.emissiveFactor = material.emissiveFactor;
            matUBO.isDisparity = material.isDisparity;
            void* data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory, 0,
                        sizeof(MaterialBufferObject), 0,
                        &data);
            memcpy(data, &matUBO, sizeof(MaterialBufferObject));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory);
        }
    }


    std::shared_ptr<MaterialInstance> SceneRenderer::initializeMaterial(
        Entity entity, const MaterialComponent& materialComponent) {
        auto materialInstance = std::make_shared<MaterialInstance>();
        if (std::filesystem::exists(materialComponent.albedoTexturePath)) {
            materialInstance->baseColorTexture = EditorUtils::createTextureFromFile(materialComponent.albedoTexturePath,
                m_context);
        }
        else {
            materialInstance->baseColorTexture = EditorUtils::createEmptyTexture(1280, 720, VK_FORMAT_R8G8B8A8_UNORM,
                m_context, true);
        }
        Log::Logger::getInstance()->info("Created Material for Entity: {}", entity.getName());
        return materialInstance;
    }


    std::shared_ptr<MeshInstance> SceneRenderer::initializeMesh(const MeshComponent& meshComponent) {
        // Load mesh data from file or other source
        if (meshComponent.m_meshPath.empty() && meshComponent.m_type != CAMERA_GIZMO)
            return nullptr;

        MeshData meshData = MeshData(meshComponent.m_type, meshComponent.m_meshPath);
        auto meshInstance = std::make_shared<MeshInstance>();
        meshInstance->topology = meshComponent.m_type == OBJ_FILE
                                     ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                     : VK_PRIMITIVE_TOPOLOGY_POINT_LIST; // Set topology based on mesh data

        meshInstance->vertexCount = meshData.vertices.size();
        meshInstance->indexCount = meshData.indices.size();
        VkDeviceSize vertexBufferSize = meshData.vertices.size() * sizeof(Vertex);
        VkDeviceSize indexBufferSize = meshData.indices.size() * sizeof(uint32_t);
        meshInstance->m_type = meshComponent.m_type;
        if (meshComponent.m_type == CAMERA_GIZMO) {
            meshInstance->vertexCount = 21;
            return meshInstance;
        }

        if (!vertexBufferSize)
            return nullptr;

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex m_DataPtr
        CHECK_RESULT(m_context->vkDevice().createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            meshData.vertices.data()))
        // Index m_DataPtr
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                meshData.indices.data()))
        }
        CHECK_RESULT(m_context->vkDevice().createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            meshInstance->vertexBuffer, vertexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Vertex",
            m_context->getDebugUtilsObjectNameFunction()));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshInstance->indexBuffer,
                indexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Index",
                m_context->getDebugUtilsObjectNameFunction()));
        }
        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_context->vkDevice().createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, meshInstance->vertexBuffer->m_buffer, 1, &copyRegion);
        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, meshInstance->indexBuffer->m_buffer, 1, &copyRegion);
        }
        m_context->vkDevice().flushCommandBuffer(copyCmd, m_context->vkDevice().m_TransferQueue, true);
        vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(m_context->vkDevice().m_LogicalDevice, vertexStaging.memory, nullptr);

        if (indexBufferSize > 0) {
            vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(m_context->vkDevice().m_LogicalDevice, indexStaging.memory, nullptr);
        }
        return meshInstance;
    }
}
