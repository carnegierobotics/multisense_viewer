//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"
#include "Editor3DLayer.h"

#include "Viewer/VkRender/Components/MaterialComponent.h"
#include "Viewer/VkRender/Editors/Common/CommonEditorFunctions.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo& createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        addUIData<Editor3DViewportUI>();


        m_descriptorRegistry.createManager(DescriptorType::Viewport3DTexture, m_context->vkDevice());
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);
        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(uuid, m_createInfo);
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());

        for (auto& frameIndex : m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                frameIndex,
                sizeof(int32_t), nullptr, "Editor3DViewport:ShaderSelectionBuffer",
                m_context->getDebugUtilsObjectNameFunction());
        }
    }

    void Editor3DViewport::onEditorResize() {
        m_editorCamera->setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);
        m_sceneRenderer->setActiveCamera(m_editorCamera);
        auto& ci = m_sceneRenderer->getCreateInfo();
        ci.width = m_createInfo.width;
        ci.height = m_createInfo.height;
        m_sceneRenderer->resize(ci);
        onRenderSettingsChanged();
    }

    void Editor3DViewport::onRenderSettingsChanged() {
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(getUUID(), m_createInfo);
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());

        if (imageUI->selectedImageType == OutputTextureImageType::Color) {
            textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        }
        else if (imageUI->selectedImageType == OutputTextureImageType::Depth) {
            textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedDepthImage;
        }

        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);
        m_activeScene = m_context->activeScene();
        m_sceneRenderer->setActiveCamera(m_editorCamera);
    }

    void Editor3DViewport::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);

        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;
        m_sceneRenderer->setActiveCamera(m_editorCamera);

        auto& e = m_context->getSelectedEntity();
        if (e && e.hasComponent<CameraComponent>()) {
            auto& camera = e.getComponent<CameraComponent>();
            if (camera.renderFromViewpoint() && imageUI->renderFromViewpoint) {
                // If the selected entity has a camera with renderFromViewpoint, use it
                m_sceneRenderer->setActiveCamera(camera.camera);
                m_lastActiveCamera = &camera; // Update the last active camera
            }
        }
        else if (m_lastActiveCamera && m_lastActiveCamera->renderFromViewpoint()) {
            // Use the last active camera if it still has renderFromViewpoint enabled
            m_sceneRenderer->setActiveCamera(m_lastActiveCamera->camera);
        }

        m_sceneRenderer->m_saveNextFrame = imageUI->saveNextFrame;


        auto frameIndex = m_context->currentFrameIndex();
        // Map and copy data to the global uniform buffer

        void* data;
        vkMapMemory(m_context->vkDevice().m_LogicalDevice,
                    m_shaderSelectionBuffer[frameIndex]->m_memory, 0, sizeof(int32_t), 0, &data);
        memcpy(data, &imageUI->depthColorOption, sizeof(int32_t));
        vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_shaderSelectionBuffer[frameIndex]->m_memory);

        m_sceneRenderer->update();
    }


    void Editor3DViewport::onRender(CommandBuffer& commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.frameIndex);

        // Render each group
        for (auto& [pipeline, commands] : renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto& command : commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }
    }

    void Editor3DViewport::collectRenderCommands(
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
        uint32_t frameIndex) {
        if (!m_meshInstances) {
            m_meshInstances = setupMesh();
            Log::Logger::getInstance()->info("Created MeshInstance for 3DViewport");
        }
        if (!m_meshInstances)
            return;

        PipelineKey key = {};
        key.setLayouts.resize(1);

        if (m_ui->resizeActive) {
            m_descriptorRegistry.getManager(DescriptorType::Viewport3DTexture).freeDescriptorSets();
        }

        // Prepare descriptor writes based on your texture or other resources
        std::array<VkWriteDescriptorSet, 2> writeDescriptors{};
        writeDescriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[0].dstBinding = 0; // Binding index
        writeDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptors[0].descriptorCount = 1;
        writeDescriptors[0].pImageInfo = &m_colorTexture->getDescriptorInfo();
        writeDescriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[1].dstBinding = 1; // Binding index
        writeDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptors[1].descriptorCount = 1;
        writeDescriptors[1].pBufferInfo = &m_shaderSelectionBuffer[frameIndex]->m_descriptorBufferInfo;
        std::vector descriptorWrites = {writeDescriptors[0], writeDescriptors[1]};
        VkDescriptorSet descriptorSet = m_descriptorRegistry.getManager(DescriptorType::Viewport3DTexture).getOrCreateDescriptorSet(descriptorWrites);
        key.setLayouts[0] = m_descriptorRegistry.getManager(DescriptorType::Viewport3DTexture).getDescriptorSetLayout();
        // Use default descriptor set layout
        key.vertexShaderName = "default2D.vert";
        key.fragmentShaderName = "default2D.frag";
        key.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        key.polygonMode = VK_POLYGON_MODE_FILL;

        std::vector<VkVertexInputBindingDescription> vertexInputBinding = {
                {0, sizeof(VkRender::ImageVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        };        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
        };
        key.vertexInputBindingDescriptions = vertexInputBinding;
        key.vertexInputAttributes = vertexInputAttributes;

        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        if (imageUI->reloadViewportShader){
            m_pipelineManager.removePipeline(key);
        }
        // Create or retrieve the pipeline
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.debugName = "Editor3DViewport::";
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[0] = descriptorSet; // Assign the descriptor set

        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void Editor3DViewport::bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command) {
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        uint32_t frameIndex = commandBuffer.frameIndex;

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

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          command.pipeline->pipeline()->getPipeline());


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

        if (command.meshInstance->indexCount > 0) {
            vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
        }
    }

    void Editor3DViewport::onMouseMove(const MouseButtons& mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera->rotate(mouse.dx, mouse.dy);
        }
    }

    void Editor3DViewport::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera->setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }

    void Editor3DViewport::onKeyCallback(const Input& input) {
        if (input.lastKeyPress == GLFW_KEY_KP_0 && input.action == GLFW_PRESS) {
            auto& e = m_context->getSelectedEntity();
            if (e && e.hasComponent<CameraComponent>()) {
                auto& camera = e.getComponent<CameraComponent>();
                camera.renderFromViewpoint() = true;
            }
        }

        if (input.lastKeyPress == GLFW_KEY_KP_1 && input.action == GLFW_PRESS) {
            m_sceneRenderer->setActiveCamera(m_editorCamera);
        }
    }


    std::shared_ptr<MeshInstance> Editor3DViewport::setupMesh() {
        std::vector<VkRender::ImageVertex> vertices = {
            // Bottom-left corner
            {glm::vec2{-1.0f, -1.0f}, glm::vec2{0.0f, 0.0f}},
            // Bottom-right corner
            {glm::vec2{1.0f, -1.0f}, glm::vec2{1.0f, 0.0f}},
            // Top-right corner
            {glm::vec2{1.0f, 1.0f}, glm::vec2{1.0f, 1.0f}},
            // Top-left corner
            {glm::vec2{-1.0f, 1.0f}, glm::vec2{0.0f, 1.0f}}
        };
        // Define the indices for two triangles that make up the quad
        std::vector<uint32_t> indices = {
            0, 1, 2, // First triangle (bottom-left to top-right)
            2, 3, 0 // Second triangle (top-right to bottom-left)
        };

        auto meshInstance = std::make_shared<MeshInstance>();

        meshInstance->vertexCount = vertices.size();
        meshInstance->indexCount = indices.size();
        VkDeviceSize vertexBufferSize = vertices.size() * sizeof(VkRender::ImageVertex);
        VkDeviceSize indexBufferSize = indices.size() * sizeof(uint32_t);

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
            vertices.data()))
        // Index m_DataPtr
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                indices.data()))
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
};
