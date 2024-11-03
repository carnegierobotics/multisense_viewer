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
        setupDescriptors();
        m_editorCamera = std::make_shared<Camera>(m_createInfo.width, m_createInfo.height);
        m_sceneRenderer = m_context->getOrAddSceneRendererByUUID(uuid, m_createInfo);
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = m_sceneRenderer->getOffscreenFramebuffer().resolvedImage;
        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        updateDescriptor(m_colorTexture->getDescriptorInfo());
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

            switch (imageUI->depthColorOption) {
            case DepthColorOption::Invert:
                break;
            case DepthColorOption::Normalize:
                break;
            case DepthColorOption::JetColormap:
                break;
            case DepthColorOption::ViridisColormap:
                break;
            }
        }

        m_colorTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        updateDescriptor(m_colorTexture->getDescriptorInfo());

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

        m_sceneRenderer->update();
    }


    void Editor3DViewport::onRender(CommandBuffer& commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> renderGroups;
        collectRenderCommands(renderGroups);

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
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups) {

        if (!m_meshInstances) {
            m_meshInstances = setupMesh();
            Log::Logger::getInstance()->info("Created MeshInstance for 3DViewport");
        }
        if (!m_meshInstances)
            return;

        PipelineKey key = {};
        key.setLayouts.push_back(m_descriptorSetLayout); // Use default descriptor set layout
        key.vertexShaderName = "default2D.vert";
        std::string shadername;
        auto imageUI = std::dynamic_pointer_cast<Editor3DViewportUI>(m_ui);
        if (imageUI->selectedImageType == OutputTextureImageType::Color) {
            shadername = "default2D.frag";
        }
        else if (imageUI->selectedImageType == OutputTextureImageType::Depth) {
            shadername = "default2DGray.frag";
        }
        key.fragmentShaderName = shadername;
        key.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        key.polygonMode = VK_POLYGON_MODE_FILL;


        VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(ImageVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
        };
        key.vertexInputBindingDescriptions = vertexInputBinding;
        key.vertexInputAttributes = vertexInputAttributes;

        // Create or retrieve the pipeline
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
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

        vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            command.pipeline->pipeline()->getPipelineLayout(),
            0, // Set 0 (entity descriptor set)
            1,
            &m_descriptorSets[frameIndex],
            0,
            nullptr
        );

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

    void Editor3DViewport::setupDescriptors() {
        uint32_t numSwapchainImages = m_context->swapChainBuffers().size();

        std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, numSwapchainImages * 2},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, numSwapchainImages * 2},
        };

        VkDescriptorPoolCreateInfo descriptorPoolCI{};
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        descriptorPoolCI.pPoolSizes = poolSizes.data();
        descriptorPoolCI.maxSets = numSwapchainImages * static_cast<uint32_t>(poolSizes.size());
        CHECK_RESULT(
            vkCreateDescriptorPool(m_context->vkDevice().m_LogicalDevice, &descriptorPoolCI, nullptr, &m_descriptorPool
            ));


        // Scene (matrices and environment maps)
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {
                    1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    nullptr
                },
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            CHECK_RESULT(
                vkCreateDescriptorSetLayout(m_context->vkDevice().m_LogicalDevice, &descriptorSetLayoutCI,
                    nullptr,
                    &m_descriptorSetLayout));

            m_descriptorSets.resize(numSwapchainImages);
            for (size_t i = 0; i < numSwapchainImages; i++) {
                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = m_descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &m_descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                VkResult res = vkAllocateDescriptorSets(m_context->vkDevice().m_LogicalDevice, &descriptorSetAllocInfo,
                                                        &m_descriptorSets[i]);
                if (res != VK_SUCCESS)
                    throw std::runtime_error("Failed to allocate descriptor sets");

            }
        }
    }

    void Editor3DViewport::updateDescriptor(const VkDescriptorImageInfo &info) {
        VkWriteDescriptorSet writeDescriptorSets{};
        for (const auto &descriptor: m_descriptorSets) {
            writeDescriptorSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets.descriptorCount = 1;
            writeDescriptorSets.dstSet = descriptor;
            writeDescriptorSets.dstBinding = 0;
            writeDescriptorSets.pImageInfo = &info;
            vkUpdateDescriptorSets(m_context->vkDevice().m_LogicalDevice, 1, &writeDescriptorSets, 0, nullptr);
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
