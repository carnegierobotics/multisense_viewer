//
// Created by mgjer on 30/09/2024.
//


#include "Viewer/VkRender/Editors/Common/SceneRenderer/SceneRenderer.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"


namespace VkRender {
    SceneRenderer::SceneRenderer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        createDescriptorPool();
        m_renderToOffscreen = true;
    }

    void SceneRenderer::onEditorResize() {
        m_activeCamera->setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);
    }

    SceneRenderer::~SceneRenderer() {
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_context->vkDevice().m_LogicalDevice, m_descriptorPool, nullptr);
        }
        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, m_descriptorSetLayout, nullptr);
        }
        if (m_materialDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, m_materialDescriptorSetLayout, nullptr);
        }
    }

    void SceneRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();
        // TODO not clear what this does but it creates render reosurces of the editor was copied as part of a split operation
        auto view = m_activeScene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto e: view) {
            onComponentAdded(Entity(e, m_activeScene.get()), view.get<MeshComponent>(e));
        }

        auto view2 = m_activeScene->getRegistry().view<MaterialComponent>();
        for (auto e: view2) {
            onComponentAdded(Entity(e, m_activeScene.get()), view2.get<MaterialComponent>(e));
        }
    }

    void SceneRenderer::onUpdate() {
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;

        for (auto &[uuid, entity]: m_createInfo.sharedUIContextData->selectedEntityMap) {
            if (!entity)
                continue;
            std::string entityTag = entity.getComponent<TagComponent>().Tag;
            if (uuid == getUUID()) {
                auto &cameraComponent = entity.getComponent<CameraComponent>();
                if (cameraComponent.renderFromViewpoint()) {
                    auto &transform = entity.getComponent<TransformComponent>();
                    cameraComponent.camera.setPerspective(
                        static_cast<float>(m_createInfo.width) / m_createInfo.height);
                    cameraComponent().pose.pos = transform.getPosition();
                    cameraComponent().pose.q = transform.getRotationQuaternion();
                    cameraComponent().updateViewMatrix();
                    transform.getPosition() = cameraComponent().pose.pos;
                    transform.getRotationQuaternion() = cameraComponent().pose.q;
                    m_activeCamera = std::make_shared<Camera>(cameraComponent.camera);
                }
            }
        }

        auto view = m_activeScene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto entity: view) {
            updateGlobalUniformBuffer(m_context->currentFrameIndex(), Entity(entity, m_activeScene.get()));
        }
        // Retrieve the list of valid entities from the active scene
        std::unordered_set<UUID> validEntities;
        for (auto entity: view) {
            validEntities.insert(Entity(entity, m_activeScene.get()).getUUID());
        }
        // Iterate over m_entityRenderData and find entities that no longer exist in the scene
        std::vector<UUID> staleEntities;
        for (const auto &[entityUUID, renderData]: m_entityRenderData) {
            if (validEntities.find(entityUUID) == validEntities.end()) {
                // If the entity UUID is not found in valid entities, mark it as stale
                staleEntities.push_back(entityUUID);
            }
        }
        // Remove each stale entity from m_entityRenderData
        for (UUID staleEntityUUID: staleEntities) {
            m_entityRenderData.erase(staleEntityUUID);
            Log::Logger::getInstance()->info("Cleaned up resources for Entity: {}",
                                             staleEntityUUID.operator std::string());
        }
    }


    void SceneRenderer::onRender(CommandBuffer &commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand> > renderGroups;
        collectRenderCommands(renderGroups);
        // Render each group
        for (auto &[pipeline, commands]: renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto &command: commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }
    }

    void SceneRenderer::bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command) {
        // Bind vertex buffers
        VkBuffer vertexBuffers[] = {command.mesh->vertexBuffer.m_Buffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer.getActiveBuffer(), 0, 1, vertexBuffers, offsets);
        // Bind index buffer if the mesh has indices
        if (command.mesh->indexBuffer.m_Buffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(commandBuffer.getActiveBuffer(), command.mesh->indexBuffer.m_Buffer, 0,
                                 VK_INDEX_TYPE_UINT32);
        }
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        uint32_t frameIndex = commandBuffer.frameIndex;
        UUID entityUUID = command.entity.getUUID();
        auto &renderData = m_entityRenderData[entityUUID];
        // Bind the entity's descriptor set (which includes both camera and model data)
        vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            command.pipeline->pipeline()->getPipelineLayout(),
            0, // Set 0 (entity descriptor set)
            1,
            &renderData.descriptorSets[frameIndex],
            0,
            nullptr
        );
        // Bind descriptor sets (e.g., material and global descriptors)
        if (command.entity.hasComponent<MaterialComponent>()) {
            vkCmdBindDescriptorSets(
                commandBuffer.getActiveBuffer(),
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                command.pipeline->pipeline()->getPipelineLayout(),
                1,
                1, // Descriptor set count
                &renderData.materialDescriptorSets[frameIndex],
                0, // Dynamic offset count
                nullptr // Dynamic offsets
            );
        }
        // Issue the draw call
        if (command.mesh->indexBuffer.m_Buffer != VK_NULL_HANDLE) {
            // Indexed draw call
            vkCmdDrawIndexed(commandBuffer.getActiveBuffer(), command.mesh->indexCount, 1, 0, 0, 0);
        } else {
            // Non-indexed draw call
            vkCmdDraw(commandBuffer.getActiveBuffer(), command.mesh->vertexCount, 1, 0, 0);
        }
    }

    void SceneRenderer::collectRenderCommands(
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand> > &renderGroups) {
        auto view = m_activeScene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto e: view) {
            Entity entity(e, m_activeScene.get());
            std::string tag = entity.getName();
            UUID uuid = entity.getUUID();
            auto &meshComponent = entity.getComponent<MeshComponent>();
            auto &transformComponent = entity.getComponent<TransformComponent>();
            // Create the pipeline key based on whether the material exists or not
            PipelineKey key = {};
            key.setLayouts.push_back(m_descriptorSetLayout); // Use default descriptor set layout
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
            } else {
                meshInstance = it->second;
            }
            key.topology = meshInstance->topology;
            key.polygonMode = meshComponent.polygonMode;
            // Check if the entity has a MaterialComponent
            std::shared_ptr<MaterialInstance> materialInstance = nullptr;
            if (entity.hasComponent<MaterialComponent>()) {
                auto &materialComponent = entity.getComponent<MaterialComponent>();
                auto materialIt = m_materialInstances.find(uuid);
                if (materialIt == m_materialInstances.end()) {
                    materialInstance = initializeMaterial(entity, materialComponent);
                    m_materialInstances[uuid] = materialInstance;
                } else {
                    materialInstance = materialIt->second;
                }
                key.vertexShaderName = materialComponent.vertexShaderName;
                key.fragmentShaderName = materialComponent.fragmentShaderName;
                key.renderMode = materialInstance->renderMode;
                key.setLayouts.push_back(m_materialDescriptorSetLayout);
                key.materialPtr = reinterpret_cast<uint64_t *>(materialInstance.get());
            }

            // Create or retrieve the pipeline
            RenderPassInfo renderPassInfo{};
            renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
            renderPassInfo.renderPass = m_renderPass->getRenderPass();
            auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
            // Create the render command
            RenderCommand command;
            command.entity = entity;
            command.pipeline = pipeline;
            command.mesh = meshInstance.get();
            command.materialInstance = materialInstance.get(); // May be null if no material is attached
            command.transform = &transformComponent;
            // Add to render group
            renderGroups[pipeline].push_back(command);
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, MeshComponent &meshComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        if (m_meshInstances.contains(entity.getUUID())) {
            m_meshInstances.erase(entity.getUUID());
        }
        m_entityRenderData[entity.getUUID()].cameraBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].modelBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].descriptorSets.resize(m_context->swapChainBuffers().size());
        // Create attachable UBO buffers and such
        for (int i = 0; i < m_context->swapChainBuffers().size(); ++i) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &m_entityRenderData[entity.getUUID()].cameraBuffer[i],
                sizeof(GlobalUniformBufferObject));
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &m_entityRenderData[entity.getUUID()].modelBuffer[i],
                sizeof(glm::mat4));
            allocatePerEntityDescriptorSet(i, entity);
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MeshComponent &meshComponent) {
        if (m_meshInstances.contains(entity.getUUID())) {
            m_meshInstances.erase(entity.getUUID());
        }
        if (m_entityRenderData.contains(entity.getUUID())) {
            m_entityRenderData[entity.getUUID()].cameraBuffer.clear();
            m_entityRenderData[entity.getUUID()].modelBuffer.clear();
            m_entityRenderData[entity.getUUID()].descriptorSets.clear();
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
                &m_entityRenderData[entity.getUUID()].materialBuffer[i],
                sizeof(MaterialBufferObject));
            updateGlobalUniformBuffer(i, entity);
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, MaterialComponent &materialComponent) {
        if (m_materialInstances.contains(entity.getUUID())) {
            m_materialInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, MaterialComponent &materialComponent) {
        if (materialComponent.updateShaders) {
            if (m_materialInstances.contains(entity.getUUID())) {
                m_materialInstances.erase(entity.getUUID());
            }
        }
    }

    void SceneRenderer::createDescriptorPool() {
        // Estimate the maximum number of entities you expect
        const uint32_t maxEntities = 1000; // Adjust based on your needs
        const uint32_t maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());
        // Total descriptor sets needed
        uint32_t descriptorCount = maxEntities * maxFramesInFlight;
        // Pool sizes for each descriptor type
        std::array<VkDescriptorPoolSize, 3> poolSizes = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = descriptorCount; // For camera buffers
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = descriptorCount; // For model buffers
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = descriptorCount; // For model buffers
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = descriptorCount;
        if (vkCreateDescriptorPool(m_context->vkDevice().m_LogicalDevice, &poolInfo, nullptr, &m_descriptorPool) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool!");
        }
        // Binding for camera buffer (Uniform Buffer)
        VkDescriptorSetLayoutBinding cameraBufferBinding = {};
        cameraBufferBinding.binding = 0;
        cameraBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cameraBufferBinding.descriptorCount = 1;
        cameraBufferBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        // Adjust based on your shader stages
        cameraBufferBinding.pImmutableSamplers = nullptr;
        // Binding for model buffer (Uniform Buffer)
        VkDescriptorSetLayoutBinding modelBufferBinding = {};
        modelBufferBinding.binding = 1;
        modelBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        modelBufferBinding.descriptorCount = 1;
        modelBufferBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        modelBufferBinding.pImmutableSamplers = nullptr;
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {cameraBufferBinding, modelBufferBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, &layoutInfo, nullptr,
                                        &m_descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
        /*** MATERIAL DESCRIPTOR SETUP ***/
        // Initialize GPU resources based on materialComponent data
        {
            std::vector<VkDescriptorSetLayoutBinding> materialBindings;
            // Uniform Buffer Binding
            VkDescriptorSetLayoutBinding uboLayoutBinding = {};
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            uboLayoutBinding.descriptorCount = 1;
            uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            uboLayoutBinding.pImmutableSamplers = nullptr;
            materialBindings.push_back(uboLayoutBinding);
            // Texture Binding (if material uses a texture)
            VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            samplerLayoutBinding.pImmutableSamplers = nullptr;
            materialBindings.push_back(samplerLayoutBinding);
            // Create Descriptor Set Layout
            VkDescriptorSetLayoutCreateInfo materialDescriptorInfo = {};
            materialDescriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            materialDescriptorInfo.bindingCount = static_cast<uint32_t>(materialBindings.size());
            materialDescriptorInfo.pBindings = materialBindings.data();
            if (vkCreateDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, &materialDescriptorInfo, nullptr,
                                            &m_materialDescriptorSetLayout) !=
                VK_SUCCESS) {
                throw std::runtime_error("Failed to create descriptor set layout!");
            }
        }
    }

    void SceneRenderer::allocatePerEntityDescriptorSet(uint32_t frameIndex, Entity entity) {
        auto &renderData = m_entityRenderData[entity.getUUID()];
        // Allocate Descriptor Set
        VkDescriptorSetLayout layouts[] = {m_descriptorSetLayout};
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(m_context->vkDevice().m_LogicalDevice, &allocInfo,
                                     &renderData.descriptorSets[frameIndex]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set for entity!");
        }
        std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
        // Write Camera Buffer
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = renderData.descriptorSets[frameIndex];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &renderData.cameraBuffer[frameIndex].m_DescriptorBufferInfo;
        // Write Model Buffer
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = renderData.descriptorSets[frameIndex];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &renderData.modelBuffer[frameIndex].m_DescriptorBufferInfo;
        vkUpdateDescriptorSets(m_context->vkDevice().m_LogicalDevice, descriptorWrites.size(),
                               descriptorWrites.data(), 0, nullptr);
    }

    void SceneRenderer::updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity) {
        // Get the active camera entity
        auto &cameraComponent = m_activeCamera;;
        // Compute view and projection matrices
        if (cameraComponent.get()) {
            GlobalUniformBufferObject globalUBO = {};
            globalUBO.view = cameraComponent->matrices.view;
            globalUBO.projection = cameraComponent->matrices.perspective;
            globalUBO.cameraPosition = cameraComponent->pose.pos;
            // Map and copy data to the global uniform buffer
            void *data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex].m_Memory, 0, sizeof(globalUBO), 0,
                        &data);
            memcpy(data, &globalUBO, sizeof(globalUBO));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex].m_Memory);
        }
        if (entity.hasComponent<TransformComponent>()) {
            void *data;
            auto &transformComponent = m_activeScene->getRegistry().get<TransformComponent>(entity);
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex].m_Memory, 0, VK_WHOLE_SIZE, 0,
                        &data);
            auto *modelMatrices = reinterpret_cast<glm::mat4 *>(data);
            *modelMatrices = transformComponent.getTransform();
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].modelBuffer[frameIndex].m_Memory);
        }
        if (entity.hasComponent<MaterialComponent>() && !m_entityRenderData[entity.getUUID()].materialBuffer.empty()) {
            auto &material = entity.getComponent<MaterialComponent>();
            MaterialBufferObject matUBO = {};
            matUBO.baseColor = material.baseColor;
            matUBO.metallic = material.metallic;
            matUBO.roughness = material.roughness;
            matUBO.emissiveFactor = material.emissiveFactor;
            void *data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex].m_Memory, 0,
                        sizeof(MaterialBufferObject), 0,
                        &data);
            memcpy(data, &matUBO, sizeof(MaterialBufferObject));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex].m_Memory);
        }
    }


    std::shared_ptr<MaterialInstance> SceneRenderer::initializeMaterial(
        Entity entity, const MaterialComponent &materialComponent) {
        // Create a new MaterialInstance
        auto materialInstance = std::make_shared<MaterialInstance>();
        auto &renderData = m_entityRenderData[entity.getUUID()];
        const auto maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());
        // load textures
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(materialComponent.albedoTexturePath.string().c_str(), &texWidth, &texHeight,
                                    &texChannels, STBI_rgb_alpha);
        auto imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * texChannels);
        if (!pixels) {
            // load empty texture
            Log::Logger::getInstance()->error("Failed to load texture image: {}. Reverting to empty",
                                              materialComponent.albedoTexturePath.string());
            pixels = stbi_load((Utils::getTexturePath() / "moon.png").string().c_str(), &texWidth, &texHeight,
                               &texChannels, STBI_rgb_alpha);
            imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * texChannels);
            if (!pixels) {
                throw std::runtime_error("Failed to load backup texture image");
            }
        }
        materialInstance->baseColorTexture = std::make_shared<Texture2D>(
            pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight), &m_context->vkDevice(),
            m_context->vkDevice().m_TransferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        stbi_image_free(pixels);
        renderData.materialDescriptorSets.resize(maxFramesInFlight);
        for (int frameIndex = 0; frameIndex < maxFramesInFlight; ++frameIndex) {
            // Allocate Descriptor Set
            VkDescriptorSetLayout layouts[] = {m_materialDescriptorSetLayout};
            VkDescriptorSetAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = m_descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = layouts;
            VkResult res = vkAllocateDescriptorSets(m_context->vkDevice().m_LogicalDevice, &allocInfo,
                                                    &renderData.materialDescriptorSets[frameIndex]);
            if (res != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate descriptor set for entity!");
            }
            std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
            // Write Camera Buffer
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = renderData.materialDescriptorSets[frameIndex];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &renderData.materialBuffer[frameIndex].m_DescriptorBufferInfo;
            // Write Model Buffer
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = renderData.materialDescriptorSets[frameIndex];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &materialInstance->baseColorTexture->m_descriptor;

            vkUpdateDescriptorSets(m_context->vkDevice().m_LogicalDevice, descriptorWrites.size(),
                                   descriptorWrites.data(), 0, nullptr);
        }

        Log::Logger::getInstance()->info("Created Material for Entity: {}", entity.getName());
        return materialInstance;
    }


    std::shared_ptr<MeshInstance> SceneRenderer::initializeMesh(const MeshComponent &meshComponent) {
        // Load mesh data from file or other source
        MeshData meshData = MeshData(meshComponent.meshPath);
        // TODO use staging buffer for static meshes
        // Create MeshInstance instance
        auto meshInstance = std::make_shared<MeshInstance>();
        meshInstance->vertexCount = meshData.vertices.size();
        meshInstance->indexCount = meshData.indices.size();
        meshInstance->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // Set topology based on mesh data
        uint32_t vertexBufferSize = meshData.vertices.size() * sizeof(Vertex);
        uint32_t indexBufferSize = meshData.indices.size() * sizeof(uint32_t);
        if (!vertexBufferSize)
            return nullptr;
        // Create vertex buffer
        m_context->vkDevice().createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &meshInstance->vertexBuffer,
            vertexBufferSize,
            meshData.vertices.data());
        // Create index buffer if the mesh has indices
        if (meshComponent.hasIndices) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &meshInstance->indexBuffer,
                indexBufferSize,
                meshData.indices.data());
        }
        return meshInstance;
    }
}
