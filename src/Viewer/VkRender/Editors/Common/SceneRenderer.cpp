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
    SceneRenderer::SceneRenderer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        createDescriptorPool();
        m_renderToOffscreen = true;

       // m_videoPlaybackSystem = std::make_shared<VideoPlaybackSystem>(m_context);
        m_activeCamera = Camera(m_createInfo.width, m_createInfo.height);

    }

    void SceneRenderer::onEditorResize() {
        m_activeCamera.setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);
    }

    SceneRenderer::~SceneRenderer() {
        m_entityRenderData.clear();

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
        for (auto entity: view) {
            updateGlobalUniformBuffer(m_context->currentFrameIndex(), Entity(entity, m_activeScene.get()));
        }
        // Update
        /*
        auto materialview = m_activeScene->getRegistry().view<MaterialComponent>();
        for (auto e: materialview) {
            Entity entity(e, m_activeScene.get());
            auto &materialComponent = entity.getComponent<MaterialComponent>();
            if (materialComponent.usesVideoSource && m_materialInstances.contains(entity.getUUID()) &&
                std::filesystem::exists(materialComponent.videoFolderSource)) {
                MaterialInstance *materialInstance = m_materialInstances[entity.getUUID()].get();
                VideoSource::ImageType imageType = VideoSource::ImageType::RGB;
                if (materialComponent.isDisparity)
                    imageType = VideoSource::ImageType::Disparity16Bit;
                // Determined based on user input or file inspection
                float fps = 30.0f;
                bool loop = true;
                m_videoPlaybackSystem->addVideoSource(materialComponent.videoFolderSource, imageType, fps, loop,
                                                      entity.getUUID());

                materialInstance->baseColorTexture = m_videoPlaybackSystem->getTexture(entity.getUUID());
                updateMaterialDescriptors(entity, materialInstance);
            }
        }
        auto pointcloudView = m_activeScene->getRegistry().view<PointCloudComponent>();
        for (auto e: pointcloudView) {
            Entity entity(e, m_activeScene.get());
            auto &pointCloudComponent = entity.getComponent<PointCloudComponent>();

            if (pointCloudComponent.usesVideoSource && m_pointCloudInstances.contains(entity.getUUID()) &&
                std::filesystem::exists(
                        pointCloudComponent.depthVideoFolderSource) && std::filesystem::exists(
                    pointCloudComponent.colorVideoFolderSource)) {
                PointCloudInstance *pointCloudInstance = m_pointCloudInstances[entity.getUUID()].get();

                // Determined based on user input or file inspection
                float fps = 30.0f;
                bool loop = true;
                m_videoPlaybackSystem->addVideoSource(pointCloudComponent.depthVideoFolderSource,
                                                      VideoSource::ImageType::Disparity16Bit, fps, loop,
                                                      entity.getUUID());

                m_videoPlaybackSystem->addVideoSource(
                        pointCloudComponent.colorVideoFolderSource, VideoSource::ImageType::RGB, fps, loop,
                        entity.getUUID());

                for (auto &texture: pointCloudInstance->textures) {
                    texture.depth = m_videoPlaybackSystem->getTexture(entity.getUUID());
                    texture.color = m_videoPlaybackSystem->getTexture(entity.getUUID());
                }
                updatePointCloudDescriptors(entity, pointCloudInstance);
            }
        }
        */
        /*
        if (imageUI->showVideoControlPanel)
            m_videoPlaybackSystem->update(m_context->deltaTime());

        if (imageUI->resetPlayback) {
            m_videoPlaybackSystem->resetAllSourcesPlayback();
        }
        */

        /*
        // TODO rethink the method for cleaning up resources
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
            if (m_entityRenderData.contains(staleEntityUUID)) {
                m_entityRenderData.erase(staleEntityUUID);
                Log::Logger::getInstance()->info("Cleaned up resources for Entity: {}",
                                                 staleEntityUUID.operator std::string());
            }
        }
        */
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
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        uint32_t frameIndex = commandBuffer.frameIndex;
        UUID entityUUID = command.entity.getUUID();
        auto &renderData = m_entityRenderData[entityUUID];

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

        // Bind the entity's descriptor set (which includes both camera and model data)
        if (command.entity.hasComponent<MeshComponent>()) {
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
        }

        // Bind descriptor sets (e.g., material and global descriptors)
        if (command.entity.hasComponent<MaterialComponent>()) {
            vkCmdBindDescriptorSets(
                    cmdBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    command.pipeline->pipeline()->getPipelineLayout(),
                    1,
                    1, // Descriptor set count
                    &renderData.materialDescriptorSets[frameIndex],
                    0, // Dynamic offset count
                    nullptr // Dynamic offsets
            );
        }
        // Bind descriptor sets for point cloud
        if (command.entity.hasComponent<PointCloudComponent>() && command.pointCloudInstance) {
            vkCmdBindDescriptorSets(
                    cmdBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    command.pipeline->pipeline()->getPipelineLayout(),
                    1,
                    1, // Descriptor set count
                    &renderData.pointCloudDescriptorSets[frameIndex],
                    0, // Dynamic offset count
                    nullptr // Dynamic offsets
            );
        }
        if (command.meshInstance->vertexBuffer) {
            // Issue the draw call
            if (command.meshInstance->indexBuffer) {
                // Indexed draw call
                vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
            } else {
                // Non-indexed draw call
                vkCmdDraw(cmdBuffer, command.meshInstance->vertexCount, 1, 0, 0);
            }
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

            std::shared_ptr<PointCloudInstance> pointCloudInstance = nullptr;
            if (entity.hasComponent<PointCloudComponent>()) {
                auto &pointCloudComponent = entity.getComponent<PointCloudComponent>();
                auto pointCLoudIt = m_pointCloudInstances.find(uuid);
                if (pointCLoudIt == m_pointCloudInstances.end()) {
                    pointCloudInstance = initializePointCloud(entity, pointCloudComponent);
                    if (!pointCloudInstance)
                        continue;
                    m_pointCloudInstances[uuid] = pointCloudInstance;
                } else {
                    pointCloudInstance = pointCLoudIt->second;
                }
                key.vertexShaderName = "pointcloud.vert.spv";
                key.fragmentShaderName = "pointcloud.frag.spv";
                key.setLayouts.push_back(m_pointCloudDescriptorSetLayout);
                key.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
                key.polygonMode = VK_POLYGON_MODE_POINT;
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
            command.meshInstance = meshInstance.get();
            command.materialInstance = materialInstance.get(); // May be null if no material is attached
            command.pointCloudInstance = pointCloudInstance.get(); // May be null if no pointcloud is attached
            command.transform = &transformComponent;
            // Add to render group
            renderGroups[pipeline].push_back(command);
        }

        // Collect pointcloud rendercommands
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
                    m_entityRenderData[entity.getUUID()].cameraBuffer[i],
                    sizeof(GlobalUniformBufferObject), nullptr, "SceneRenderer:MeshComponent:Camera",
                    m_context->getDebugUtilsObjectNameFunction());
            m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    m_entityRenderData[entity.getUUID()].modelBuffer[i],
                    sizeof(glm::mat4), nullptr, "SceneRenderer:MeshComponent:Model",
                    m_context->getDebugUtilsObjectNameFunction());
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

        if (materialComponent.reloadShader) {
            if (m_materialInstances.contains(entity.getUUID())) {
                m_materialInstances.erase(entity.getUUID());
            }
        }
    }

    void SceneRenderer::onComponentAdded(Entity entity, PointCloudComponent &pointCloudComponent) {
        // Check if I readd a meshcomponent then we should destroy the renderresources attached to it:
        m_entityRenderData[entity.getUUID()].cameraBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].modelBuffer.resize(m_context->swapChainBuffers().size());
        m_entityRenderData[entity.getUUID()].descriptorSets.resize(m_context->swapChainBuffers().size());
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
            allocatePerEntityDescriptorSet(i, entity);
        }
    }

    void SceneRenderer::onComponentRemoved(Entity entity, PointCloudComponent &pointCloudComponent) {
        if (m_pointCloudInstances.contains(entity.getUUID())) {
            m_pointCloudInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::onComponentUpdated(Entity entity, PointCloudComponent &pointCloudComponent) {
        if (m_pointCloudInstances.contains(entity.getUUID())) {
            // TODO look into reuse instead of re-creation
            m_pointCloudInstances.erase(entity.getUUID());
        }
    }

    void SceneRenderer::createDescriptorPool() {
        // Estimate the maximum number of entities you expect
        const uint32_t maxEntities = 1000; // Adjust based on your needs
        const uint32_t maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());
        // Total descriptor sets needed
        uint32_t descriptorCount = maxEntities * maxFramesInFlight;
        // Pool sizes for each descriptor type
        std::array<VkDescriptorPoolSize, 4> poolSizes = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = descriptorCount; // For camera buffers
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = descriptorCount; // For model buffers
        // Color map sampler (fragment shader)
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = descriptorCount; // Color map sampler
        // Chroma U and V samplers (fragment shader)
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[3].descriptorCount = descriptorCount * 2; // For chromaU and chromaV samplers
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
        } /*** POINTCLOUD DESCRIPTOR SETUP ***/
        // Initialize GPU resources based on materialComponent data
        {
            std::array<VkDescriptorSetLayoutBinding, 5> layoutBindings = {
                    {
                            // PointCloudParam in both vertex and fragment shader
                            {
                                    .binding = 0,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                    .descriptorCount = 1,
                                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                                  VK_SHADER_STAGE_FRAGMENT_BIT, // Used in both shaders
                                    .pImmutableSamplers = nullptr
                            },
                            // Depth map in vertex shader
                            {
                                    .binding = 1,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    .descriptorCount = 1,
                                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, // Only in vertex shader
                                    .pImmutableSamplers = nullptr
                            },
                            // Sampler color map in fragment shader
                            {
                                    .binding = 2,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    .descriptorCount = 1,
                                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, // Only in fragment shader
                                    .pImmutableSamplers = nullptr
                            },
                            // Chroma U map in fragment shader
                            {
                                    .binding = 3,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    .descriptorCount = 1,
                                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, // Only in fragment shader
                                    .pImmutableSamplers = nullptr
                            },
                            // Chroma V map in fragment shader
                            {
                                    .binding = 4,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                    .descriptorCount = 1,
                                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, // Only in fragment shader
                                    .pImmutableSamplers = nullptr
                            }
                    }
            };

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
            layoutInfo.pBindings = layoutBindings.data();
            if (vkCreateDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, &layoutInfo, nullptr,
                                            &m_pointCloudDescriptorSetLayout) !=
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
        descriptorWrites[0].pBufferInfo = &renderData.cameraBuffer[frameIndex]->m_descriptorBufferInfo;
        // Write Model Buffer
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = renderData.descriptorSets[frameIndex];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &renderData.modelBuffer[frameIndex]->m_descriptorBufferInfo;
        vkUpdateDescriptorSets(m_context->vkDevice().m_LogicalDevice, descriptorWrites.size(),
                               descriptorWrites.data(), 0, nullptr);
    }

    void SceneRenderer::updateGlobalUniformBuffer(uint32_t frameIndex, Entity entity) {
        // Get the active camera entity
        // Compute view and projection matrices
        if (entity.hasComponent<MeshComponent>()) {
            GlobalUniformBufferObject globalUBO = {};
            globalUBO.view = m_activeCamera.matrices.view;
            globalUBO.projection = m_activeCamera.matrices.perspective;
            globalUBO.cameraPosition = m_activeCamera.pose.pos;
            // Map and copy data to the global uniform buffer
            void *data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].cameraBuffer[frameIndex]->m_memory, 0, sizeof(globalUBO),
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
        if (entity.hasComponent<MaterialComponent>() && !m_entityRenderData[entity.getUUID()].materialBuffer.empty()) {
            auto &material = entity.getComponent<MaterialComponent>();
            MaterialBufferObject matUBO = {};
            matUBO.baseColor = material.baseColor;
            matUBO.metallic = material.metallic;
            matUBO.roughness = material.roughness;
            matUBO.emissiveFactor = material.emissiveFactor;
            matUBO.isDisparity = material.isDisparity;
            void *data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory, 0,
                        sizeof(MaterialBufferObject), 0,
                        &data);
            memcpy(data, &matUBO, sizeof(MaterialBufferObject));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].materialBuffer[frameIndex]->m_memory);
        }

        if (entity.hasComponent<PointCloudComponent>() && !m_entityRenderData[entity.getUUID()].pointCloudBuffer.
                empty()) {
            auto &pointCloud = entity.getComponent<PointCloudComponent>();
            PointCloudUBO pointCloudUBO = {};
            pointCloudUBO.Q = glm::mat4(1.0f);

            // Hardcoded intrinsics
            float fx = 644.5986328125f;
            float fy = 644.384033203125f;
            float cx = 466.68487548828125f;
            float cy = 306.7324523925781f;
            // Hardcoded translation in stereo rectification
            float tx = -0.27f;
            // Populate the Q matrix
            glm::mat4 Q(0.0f); // Start with an empty matrix

            Q[0][0] = fy * tx; // Q[0][0] = fy * tx
            Q[1][1] = fx * tx; // Q[1][1] = fx * tx
            Q[2][3] = -fy; // Q[2][3] = -fy
            Q[3][0] = -fy * cx * tx; // Q[3][0] = -fy * cx * tx
            Q[3][1] = -fx * cy * tx; // Q[3][1] = -fx * cy * tx
            Q[3][2] = fx * fy * tx; // Q[3][2] = fx * fy * tx
            Q[3][3] = 0.0f; // Since the cameras are rectified, no need for dcx

            pointCloudUBO.Q = Q;

            // Set the intrinsics matrix
            glm::mat4 intrinsics = glm::mat4(1.0f); // Identity matrix
            intrinsics[0][0] = 600.0f; // fx from "P"
            intrinsics[1][1] = 600.0f; // fy from "P"
            intrinsics[0][2] = 480.0f; // cx from "P"
            intrinsics[1][2] = 300.0f; // cy from "P"
            pointCloudUBO.intrinsics = intrinsics;

            // Hardcode the extrinsics matrix (R) - since the camera is stereo rectified,
            // we'll assume the translation is minimal and focus on the rotation
            glm::mat4 extrinsics = glm::mat4(1.0f); // Identity matrix

            // Rotation matrix from the JSON (R matrix)
            //extrinsics[0][0] = 0.9999923706054688f; // R00
            //extrinsics[0][1] = 0.002553278347477317f; // R01
            //extrinsics[0][2] = 0.0029585757292807102f; // R02
            //extrinsics[1][0] = -0.0025533579755574465f; // R10
            //extrinsics[1][1] = 0.9999967217445374f; // R11
            //extrinsics[1][2] = 2.3196893380372785e-05f; // R12
            //extrinsics[2][0] = -0.0029585070442408323f; // R20
            //extrinsics[2][1] = -3.075101994909346e-05f; // R21
            //extrinsics[2][2] = 0.9999956488609314f; // R22

            extrinsics[3][0] = 20.049549102783203f / 600; // Translation along X

            // Since this is stereo rectified from the left camera's viewpoint, you might
            // need to apply a small translation to account for the stereo baseline, if required
            // (translation can be added here based on the setup)

            pointCloudUBO.extrinsics = extrinsics;
            pointCloudUBO.width = 960.0f;
            pointCloudUBO.height = 600.0f;
            pointCloudUBO.disparity = 255.0f;
            pointCloudUBO.pointSize = pointCloud.pointSize;
            pointCloudUBO.hasSampler = 1.0f;
            pointCloudUBO.useColor = 0.0f;
            pointCloudUBO.scale = 1.0f;
            pointCloudUBO.focalLength = -0.27f;

            void *data;
            vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                        m_entityRenderData[entity.getUUID()].pointCloudBuffer[frameIndex]->m_memory, 0,
                        sizeof(PointCloudUBO), 0, &data);
            memcpy(data, &pointCloudUBO, sizeof(PointCloudUBO));
            vkUnmapMemory(m_context->vkDevice().m_LogicalDevice,
                          m_entityRenderData[entity.getUUID()].pointCloudBuffer[frameIndex]->m_memory);
        }
    }


    std::shared_ptr<MaterialInstance> SceneRenderer::initializeMaterial(
            Entity entity, MaterialComponent &materialComponent) {
        // Create a new MaterialInstance
        auto materialInstance = std::make_shared<MaterialInstance>();
        auto &renderData = m_entityRenderData[entity.getUUID()];
        const auto maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());

        materialInstance->baseColorTexture = EditorUtils::createTextureFromFile(
                Utils::getTexturePath() / "moon.png", m_context);

        renderData.materialDescriptorSets.resize(maxFramesInFlight);
        Log::Logger::getInstance()->info("Created Material for Entity: {}", entity.getName());
        updateMaterialDescriptors(entity, materialInstance.get());
        return materialInstance;
    }

    void SceneRenderer::updateMaterialDescriptors(
            Entity entity, MaterialInstance *materialInstance) {
        // Create a new MaterialInstance
        auto &renderData = m_entityRenderData[entity.getUUID()];
        const auto maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());

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
            descriptorWrites[0].pBufferInfo = &renderData.materialBuffer[frameIndex]->m_descriptorBufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = renderData.materialDescriptorSets[frameIndex];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &materialInstance->baseColorTexture->getDescriptorInfo();

            vkUpdateDescriptorSets(m_context->vkDevice().m_LogicalDevice, descriptorWrites.size(),
                                   descriptorWrites.data(), 0, nullptr);
        }

        Log::Logger::getInstance()->info("Updated Material descriptors for Entity: {}", entity.getName());
    }


    std::shared_ptr<PointCloudInstance> SceneRenderer::initializePointCloud(
            Entity entity, PointCloudComponent &pointCloudComponent) {
        // Create a new MaterialInstance
        // Initialize point cloud instance textures etc..
        //
        /*if (pointCloudComponent.usesVideoSource) {
            std::filesystem::path folderPath = pointCloudComponent.videoFolderSource;
            std::vector<std::filesystem::path> files;
            // Check if the folder exists
            if (std::filesystem::exists(folderPath) && std::filesystem::is_directory(folderPath)) {
                // Iterate through the folder and collect files
                for (const auto &entry: std::filesystem::directory_iterator(folderPath)) {
                    if (entry.is_regular_file()) {
                        files.push_back(entry.path());
                    }
                }
                // Sort files by filename assuming filenames are timestamps in nanoseconds
                std::sort(files.begin(), files.end(),
                          [](const std::filesystem::path &a, const std::filesystem::path &b) {
                              return a.filename().string() < b.filename().string();
                          });
                pointCloudComponent.videoFileNames = files;
                pointCloudInstance->textures.resize(maxFramesInFlight);
                for (int frameIndex = 0; frameIndex < maxFramesInFlight; ++frameIndex) {
                    int texWidth, texHeight, texChannels;
                    stbi_us *pixels = stbi_load_16(files.front().string().c_str(), &texWidth, &texHeight, &texChannels,
                                                   STBI_grey);
                    VkDeviceSize imageSize = texWidth * texHeight * texChannels * sizeof(stbi_us);
                    // Assuming STBI_rgb_alpha gives us 4 channels per pixel
                    if (!pixels) {
                        throw std::runtime_error("Failed to load backup texture image");
                    }

                    VkImageCreateInfo imageCI = Populate::imageCreateInfo();
                    imageCI.imageType = VK_IMAGE_TYPE_2D;
                    imageCI.format = VK_FORMAT_R16_UNORM;
                    imageCI.extent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};
                    imageCI.mipLevels = 1;
                    imageCI.arrayLayers = 1;
                    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
                    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
                    imageCI.usage =
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
                    imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
                    imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    imageViewCI.format = VK_FORMAT_R16_UNORM;
                    imageViewCI.subresourceRange.baseMipLevel = 0;
                    imageViewCI.subresourceRange.levelCount = 1;
                    imageViewCI.subresourceRange.baseArrayLayer = 0;
                    imageViewCI.subresourceRange.layerCount = 1;
                    imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

                    VulkanImageCreateInfo vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCI,
                                                                imageViewCI);
                    vulkanImageCreateInfo.debugInfo = "Color texture: Image Editor";
                    VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
                    textureCreateInfo.image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
                    auto texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

                    // Copy data to texturere
                    texture->loadImage(pixels, imageSize);
                    // Free the image data
                    stbi_image_free(pixels);
                    pointCloudInstance->textures[frameIndex].depth = texture;

                    pointCloudInstance->textures[frameIndex].color = EditorUtils::createTextureFromFile(
                        files.front(), m_context);
                    pointCloudInstance->textures[frameIndex].chromaU = EditorUtils::createTextureFromFile(
                        files.front(), m_context);
                    pointCloudInstance->textures[frameIndex].chromaV = EditorUtils::createTextureFromFile(
                        files.front(), m_context);
                }
            } else {
                return nullptr;
            }
        } else {
            return nullptr;
        }
*/

        auto pointCloudInstance = std::make_shared<PointCloudInstance>();
        auto &renderData = m_entityRenderData[entity.getUUID()];
        const auto maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());
        renderData.pointCloudDescriptorSets.resize(maxFramesInFlight);
        Log::Logger::getInstance()->info("Created Point Cloud Instance for Entity: {}", entity.getName());
        pointCloudInstance->textures.resize(maxFramesInFlight);
        for (int frameIndex = 0; frameIndex < maxFramesInFlight; ++frameIndex) {
            pointCloudInstance->textures[frameIndex].depth = EditorUtils::createTextureFromFile("", m_context);
            pointCloudInstance->textures[frameIndex].color = EditorUtils::createTextureFromFile("", m_context);
            pointCloudInstance->textures[frameIndex].chromaU = EditorUtils::createTextureFromFile("", m_context);
            pointCloudInstance->textures[frameIndex].chromaV = EditorUtils::createTextureFromFile("", m_context);
        }
        updatePointCloudDescriptors(entity, pointCloudInstance.get());
        return pointCloudInstance;
    }

    void SceneRenderer::updatePointCloudDescriptors(Entity entity, PointCloudInstance *pointCloudInstance) {
        auto &renderData = m_entityRenderData[entity.getUUID()];
        const auto maxFramesInFlight = static_cast<uint32_t>(m_context->swapChainBuffers().size());
        for (int frameIndex = 0; frameIndex < maxFramesInFlight; ++frameIndex) {
            // Create a descriptor set allocation info structure
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = m_descriptorPool; // Use the descriptor pool created earlier
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_pointCloudDescriptorSetLayout; // Use the pre-created descriptor set layout
            // Allocate the descriptor set
            if (vkAllocateDescriptorSets(m_context->vkDevice().m_LogicalDevice, &allocInfo,
                                         &renderData.pointCloudDescriptorSets[frameIndex]) !=
                VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate descriptor sets!");
            }
            VkDescriptorSet descriptorSet = renderData.pointCloudDescriptorSets[frameIndex];
            VkWriteDescriptorSet pointCloudWrite{};
            pointCloudWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            pointCloudWrite.dstSet = descriptorSet;
            pointCloudWrite.dstBinding = 0; // Corresponds to binding 0 in your shaders
            pointCloudWrite.dstArrayElement = 0;
            pointCloudWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            pointCloudWrite.descriptorCount = 1;
            pointCloudWrite.pBufferInfo = &renderData.pointCloudBuffer[frameIndex]->
                    m_descriptorBufferInfo;

            VkWriteDescriptorSet depthMapWrite{};
            depthMapWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            depthMapWrite.dstSet = descriptorSet;
            depthMapWrite.dstBinding = 1; // Corresponds to binding 1 in vertex shader
            depthMapWrite.dstArrayElement = 0;
            depthMapWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            depthMapWrite.descriptorCount = 1;
            depthMapWrite.pImageInfo = &pointCloudInstance->textures[frameIndex].depth->getDescriptorInfo();

            VkWriteDescriptorSet colorMapWrite{};
            colorMapWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            colorMapWrite.dstSet = descriptorSet;
            colorMapWrite.dstBinding = 2; // Corresponds to binding 2 in fragment shader
            colorMapWrite.dstArrayElement = 0;
            colorMapWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            colorMapWrite.descriptorCount = 1;
            colorMapWrite.pImageInfo = &pointCloudInstance->textures[frameIndex].color->getDescriptorInfo();

            VkWriteDescriptorSet chromaUWrite{};
            chromaUWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            chromaUWrite.dstSet = descriptorSet;
            chromaUWrite.dstBinding = 3; // Corresponds to binding 2 in fragment shader
            chromaUWrite.dstArrayElement = 0;
            chromaUWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            chromaUWrite.descriptorCount = 1;
            chromaUWrite.pImageInfo = &pointCloudInstance->textures[frameIndex].chromaU->getDescriptorInfo();

            VkWriteDescriptorSet chromaVWrite{};
            chromaVWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            chromaVWrite.dstSet = descriptorSet;
            chromaVWrite.dstBinding = 4; // Corresponds to binding 2 in fragment shader
            chromaVWrite.dstArrayElement = 0;
            chromaVWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            chromaVWrite.descriptorCount = 1;
            chromaVWrite.pImageInfo = &pointCloudInstance->textures[frameIndex].chromaV->getDescriptorInfo();

            // Group all the write descriptor sets together
            std::array<VkWriteDescriptorSet, 5> descriptorWrites = {
                    pointCloudWrite, depthMapWrite, colorMapWrite, chromaUWrite, chromaVWrite
            };

            // Update the descriptor sets
            vkUpdateDescriptorSets(m_context->vkDevice().m_LogicalDevice,
                                   static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
            // Allocate
        }
    }

    std::shared_ptr<MeshInstance> SceneRenderer::initializeMesh(const MeshComponent &meshComponent) {
        // Load mesh data from file or other source
        MeshData meshData = MeshData(meshComponent.meshDataType, meshComponent.meshPath);
        // TODO use staging buffer for static meshes
        // Create MeshInstance instance

        auto meshInstance = std::make_shared<MeshInstance>();
        meshInstance->topology = meshComponent.meshDataType == OBJ_FILE
                                 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                                 : VK_PRIMITIVE_TOPOLOGY_POINT_LIST; // Set topology based on mesh data

        meshInstance->vertexCount = meshData.vertices.size();
        meshInstance->indexCount = meshData.indices.size();
        uint32_t vertexBufferSize = meshData.vertices.size() * sizeof(Vertex);
        uint32_t indexBufferSize = meshData.indices.size() * sizeof(uint32_t);
        if (!vertexBufferSize)
            return nullptr;
        // Create vertex buffer
        m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                meshInstance->vertexBuffer,
                vertexBufferSize,
                meshData.vertices.data(), "SceneRenderer:InitializeMesh:Vertex",
                m_context->getDebugUtilsObjectNameFunction());
        // Create index buffer if the mesh has indices
        if (indexBufferSize) {
            m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    meshInstance->indexBuffer,
                    indexBufferSize,
                    meshData.indices.data(), "SceneRenderer:InitializeMesh:Index",
                    m_context->getDebugUtilsObjectNameFunction());
        }
        return meshInstance;
    }

}
