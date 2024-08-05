/**
 * @file: MultiSense-Viewer/src/Renderer/Renderer.cpp
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-4-9, mgjerde@carnegierobotics.com, Created file.
 **/

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Core/UUID.h"

#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"
#include "Viewer/Tools/Macros.h"

#include "Viewer/VkRender/Components/Components.h"


#include "Viewer/VkRender/Editors/EditorDefinitions.h"
#include "Viewer/VkRender/Core/VulkanResourceManager.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"

namespace VkRender {
    Renderer::Renderer(const std::string &title) : VulkanRenderer(title) {
        RendererConfig &config = RendererConfig::getInstance();
        this->m_title = title;
        Log::Logger::getInstance()->setLogLevel(config.getLogLevel());
        m_logger = Log::Logger::getInstance();
        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        VulkanResourceManager::getInstance(m_vulkanDevice, m_allocator);
        m_guiResources = std::make_shared<GuiResources>(m_vulkanDevice);
        backendInitialized = true;

        // Create default camera object
        createNewCamera(m_selectedCameraTag, m_width, m_height);
        m_logger->info("Initialized Backend");
        config.setGpuDevice(physicalDevice);
        // Start up usage monitor
        m_usageMonitor = std::make_shared<UsageMonitor>();
        m_usageMonitor->loadSettingsFromFile();
        m_usageMonitor->userStartSession(rendererStartTime);
        m_cameras[m_selectedCameraTag].setType(Camera::arcball);
        m_cameras[m_selectedCameraTag].setPerspective(60.0f,
                                                      static_cast<float>(m_width) / static_cast<float>(m_height));
        m_cameras[m_selectedCameraTag].resetPosition();
        // m_renderUtils.renderPass = &renderPass;
        createColorResources();
        createDepthStencil();
        createMainRenderPass();

        // Initialize shared data across editors:

        m_sharedContextData.multiSenseRendererBridge = std::make_shared<MultiSense::MultiSenseRendererBridge>();
        m_sharedContextData.multiSenseRendererGigEVisionBridge = std::make_shared<
                MultiSense::MultiSenseRendererGigEVisionBridge>();


        VulkanRenderPassCreateInfo mainMenuEditor(m_frameBuffers.data(), m_guiResources, this, &m_sharedContextData);
        mainMenuEditor.appHeight = static_cast<int32_t>(m_height);
        mainMenuEditor.appWidth = static_cast<int32_t>(m_width);
        mainMenuEditor.height = static_cast<int32_t>(m_height);
        mainMenuEditor.width = static_cast<int32_t>(m_width);
        mainMenuEditor.borderSize = 0;
        mainMenuEditor.editorTypeDescription = EditorType::None;
        mainMenuEditor.resizeable = false;
        mainMenuEditor.msaaSamples = msaaSamples;
        mainMenuEditor.swapchainImageCount = swapchain->imageCount;
        mainMenuEditor.swapchainColorFormat = swapchain->colorFormat;
        mainMenuEditor.depthFormat = depthFormat;
        m_mainEditor = std::make_unique<Editor>(mainMenuEditor);
        m_mainEditor->addUI("DebugWindow");
        m_mainEditor->addUI("MenuLayer");
        m_mainEditor->addUI("MainContextLayer");

        m_editorFactory = std::make_unique<EditorFactory>(
                VulkanRenderPassCreateInfo(m_frameBuffers.data(), m_guiResources, this, &m_sharedContextData));

        loadEditorSettings(Utils::getMultiSenseViewerProjectConfig());

        if (m_editors.empty()) {
            // add a dummy editor to get started
            auto sizeLimits = m_mainEditor->getSizeLimits();
            VulkanRenderPassCreateInfo otherEditorInfo(m_frameBuffers.data(), m_guiResources, this,
                                                       &m_sharedContextData);
            otherEditorInfo.appHeight = static_cast<int32_t>(m_height);
            otherEditorInfo.appWidth = static_cast<int32_t>(m_width);
            otherEditorInfo.borderSize = 5;
            otherEditorInfo.height = static_cast<int32_t>(m_height) - sizeLimits.MENU_BAR_HEIGHT; //- 100;
            otherEditorInfo.width = static_cast<int32_t>(m_width); //- 200;
            otherEditorInfo.x = 0; //+ 100;
            otherEditorInfo.y = sizeLimits.MENU_BAR_HEIGHT; //+ 050;
            otherEditorInfo.editorIndex = m_editors.size();
            otherEditorInfo.editorTypeDescription = EditorType::TestWindow;
            otherEditorInfo.msaaSamples = msaaSamples;
            otherEditorInfo.swapchainImageCount = swapchain->imageCount;
            otherEditorInfo.swapchainColorFormat = swapchain->colorFormat;
            otherEditorInfo.depthFormat = depthFormat;
            std::array<VkClearValue, 3> clearValues{};
            otherEditorInfo.uiContext = getMainUIContext();
            auto editor = createEditor(otherEditorInfo);
            m_editors.push_back(std::move(editor));
        }

        // Load scenes

        m_scene = std::make_unique<DefaultScene>(*this);


    }

    // TODO This should actually be handled by RendererConfig. This class handles everything saving and loading config files
    void Renderer::loadEditorSettings(const std::filesystem::path &filePath) {
        std::ifstream inFile(filePath);
        if (!inFile.is_open()) {
            Log::Logger::getInstance()->error("Failed to open file for reading: {}", filePath.string());
            return;
        }

        nlohmann::json jsonContent;
        inFile >> jsonContent;
        inFile.close();

        Log::Logger::getInstance()->info("Successfully read editor settings from: {}", filePath.string());

        if (jsonContent.contains("editors")) {
            for (const auto &jsonEditor: jsonContent["editors"]) {
                VulkanRenderPassCreateInfo createInfo(m_frameBuffers.data(), m_guiResources, this,
                                                      &m_sharedContextData);

                int32_t mainMenuBarOffset = 0;
                int32_t width = std::round(jsonEditor.value("width", 0.0) / 100 * m_width);
                int32_t height = std::round(jsonEditor.value("height", 0.0) / 100 * m_height);
                int32_t offsetX = std::round(jsonEditor.value("x", 0.0) / 100 * m_width);
                int32_t offsetY = std::round(jsonEditor.value("y", 0.0) / 100 * m_height);

                mainMenuBarOffset = offsetY == 0 ? 25 : 0;

                createInfo.x = offsetX;
                createInfo.y = offsetY;
                createInfo.y += mainMenuBarOffset;

                createInfo.width = width;
                createInfo.height = height - mainMenuBarOffset;

                if (createInfo.width + createInfo.x > m_width) {
                    createInfo.width = m_width - createInfo.x;
                }

                if (createInfo.height + createInfo.y > m_height) {
                    createInfo.height = m_height - createInfo.y;
                }

                createInfo.appWidth = m_width;
                createInfo.appHeight = m_height;


                createInfo.borderSize = jsonEditor.value("borderSize", 5);
                createInfo.editorTypeDescription = stringToEditorType(jsonEditor.value("editorTypeDescription", ""));
                createInfo.resizeable = jsonEditor.value("resizeable", true);
                createInfo.editorIndex = jsonEditor.value("editorIndex", 0);
                createInfo.uiContext = getMainUIContext();

                createInfo.msaaSamples = msaaSamples;
                createInfo.swapchainImageCount = swapchain->imageCount;
                createInfo.swapchainColorFormat = swapchain->colorFormat;
                createInfo.depthFormat = depthFormat;

                if (jsonEditor.contains("uiLayers") && jsonEditor["uiLayers"].is_array()) {
                    createInfo.uiLayers = jsonEditor["uiLayers"].get<std::vector<std::string> >();
                } else {
                    createInfo.uiLayers.clear();
                }
                // Create an Editor object with the createInfo
                m_editors.push_back(std::move(createEditor(createInfo)));

                Log::Logger::getInstance()->info("Loaded editor {}: type = {}, x = {}, y = {}, width = {}, height = {}",
                                                 createInfo.editorIndex,
                                                 editorTypeToString(createInfo.editorTypeDescription), createInfo.x,
                                                 createInfo.y, createInfo.width, createInfo.height);
            }
        }
    }

    void Renderer::createMainRenderPass() {
        VulkanRenderPassCreateInfo renderPassCreateInfo(nullptr, m_guiResources, this, &m_sharedContextData);
        renderPassCreateInfo.appHeight = static_cast<int32_t>(m_height);
        renderPassCreateInfo.appWidth = static_cast<int32_t>(m_width);
        renderPassCreateInfo.height = static_cast<int32_t>(m_height);
        renderPassCreateInfo.width = static_cast<int32_t>(m_width);
        renderPassCreateInfo.x = 0;
        renderPassCreateInfo.y = 0;
        renderPassCreateInfo.borderSize = 0;
        renderPassCreateInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        renderPassCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        renderPassCreateInfo.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        renderPassCreateInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        renderPassCreateInfo.editorTypeDescription = EditorType::None;
        renderPassCreateInfo.msaaSamples = msaaSamples;
        renderPassCreateInfo.swapchainImageCount = swapchain->imageCount;
        renderPassCreateInfo.swapchainColorFormat = swapchain->colorFormat;
        renderPassCreateInfo.depthFormat = depthFormat;
        // Start timingm_mainRenderPasses UI render pass setup
        auto startUIRenderPassSetup = std::chrono::high_resolution_clock::now();
        m_mainRenderPass = std::make_shared<VulkanRenderPass>(renderPassCreateInfo);

        std::array<VkImageView, 3> frameBufferAttachments{};
        frameBufferAttachments[0] = m_colorImage.view;
        frameBufferAttachments[1] = m_depthStencil.view;
        VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(m_width,
                                                                                        m_height,
                                                                                        frameBufferAttachments.data(),
                                                                                        frameBufferAttachments.size(),
                                                                                        m_mainRenderPass->getRenderPass());
        // TODO verify if this is ok?
        m_frameBuffers.resize(swapchain->imageCount);
        for (uint32_t i = 0; i < m_frameBuffers.size(); i++) {
            auto startFramebufferCreation = std::chrono::high_resolution_clock::now();
            frameBufferAttachments[2] = swapChainBuffers()[i].view;
            VkResult result = vkCreateFramebuffer(device, &frameBufferCreateInfo,
                                                  nullptr, &m_frameBuffers[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
        }
        m_logger->info("Prepared Renderer");
    }

    void Renderer::addDeviceFeatures() {
        if (deviceFeatures.fillModeNonSolid) {
            enabledFeatures.fillModeNonSolid = VK_TRUE;
            // Wide lines must be present for line m_Width > 1.0f
            if (deviceFeatures.wideLines) {
                enabledFeatures.wideLines = VK_TRUE;
            }
            if (deviceFeatures.samplerAnisotropy) {
                enabledFeatures.samplerAnisotropy = VK_TRUE;
            }
        }
    }

    void Renderer::buildCommandBuffers() {
        VkCommandBufferBeginInfo cmdBufInfo = Populate::commandBufferBeginInfo();
        cmdBufInfo.flags = 0;
        cmdBufInfo.pInheritanceInfo = nullptr;
        std::array<VkClearValue, 3> clearValues{};
        clearValues[0] = {{{0.1f, 0.1f, 0.1f, 1.0f}}};
        clearValues[1].depthStencil = {1.0f, 0};
        clearValues[2] = {{{0.1f, 0.1f, 0.1f, 1.0f}}};
        vkBeginCommandBuffer(drawCmdBuffers.buffers[currentFrame], &cmdBufInfo);
        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = m_mainRenderPass->getRenderPass();
        // Increase reference count by 1 here?
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = m_width;
        renderPassBeginInfo.renderArea.extent.height = m_height;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.framebuffer = m_frameBuffers[imageIndex];
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);
        for (auto &editor: m_editors) {
            editor->render(drawCmdBuffers);
        }
        m_mainEditor->render(drawCmdBuffers);
        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 1;
        Utils::setImageLayout(drawCmdBuffers.buffers[currentFrame], swapchain->buffers[imageIndex].image,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, subresourceRange,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        vkEndCommandBuffer(drawCmdBuffers.buffers[currentFrame]);
        /*
        for (auto [entity, skybox, gltfComponent]: m_registry.view<VkRender::SkyboxGraphicsPipelineComponent, GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            skybox.draw(&drawCmdBuffers, currentFrame);
            gltfComponent.model->draw(drawCmdBuffers.buffers[currentFrame]);
        }

        for (auto [entity, resources, gltfComponent]: m_registry.view<VkRender::DefaultPBRGraphicsPipelineComponent, GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            if (!resources.markedForDeletion)
                resources.draw(&drawCmdBuffers, currentFrame, gltfComponent);
            else
                resources.resources[currentFrame].busy = false;
        }


        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<DefaultGraphicsPipelineComponent2>(
                entt::exclude<DeleteComponent, ImageViewComponent, SecondaryRenderViewComponent>)) {
            auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
            resources.draw(&drawCmdBuffers);
        }

        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<CameraGraphicsPipelineComponent>(entt::exclude<DeleteComponent>)) {
            auto &resources = m_registry.get<CameraGraphicsPipelineComponent>(entity);
            resources.draw(&drawCmdBuffers);

        }

        for (auto [entity, resource]: m_registry.view<CustomModelComponent>(entt::exclude<DeleteComponent>).each()) {
            resource.draw(&drawCmdBuffers);
        }


        for (auto entity: m_registry.view<ImageViewComponent>(
                entt::exclude<DeleteComponent>)) {
            auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
            resources.draw(&drawCmdBuffers);

        }

        // Render object


        // Define the viewport with the adjusted dimensions
        VkViewport viewport{};
        VkRect2D scissor;
        if (m_guiManager->handles.fixAspectRatio) {
            // Original aspect ratio
            float mainAspectRatio = static_cast<float>(m_Width) / static_cast<float>(m_Height);
            // Sub-window dimensions (initial)
            float subWindowWidth = static_cast<float>(m_Width) - m_guiManager->handles.info->sidebarWidth;
            auto subWindowHeight = static_cast<float>(m_Height);

            if (m_guiManager->handles.enableSecondaryView) {
                mainAspectRatio = static_cast<float>(m_Width) / static_cast<float>(m_Height / 2);
                // Sub-window dimensions (initial)
                subWindowWidth = static_cast<float>(m_Width) - m_guiManager->handles.info->sidebarWidth;
                subWindowHeight = static_cast<float>(m_Height / 2);
            }
            // Calculate sub-window aspect ratio
            float subWindowAspectRatio = subWindowWidth / subWindowHeight;
            // Adjust sub-window dimensions to maintain the original aspect ratio
            if (subWindowAspectRatio > mainAspectRatio) {
                // Sub-window is wider than the main viewport
                subWindowWidth = subWindowHeight * mainAspectRatio;
            } else {
                // Sub-window is taller than the main viewport
                subWindowHeight = subWindowWidth / mainAspectRatio;
            }

            viewport = Populate::viewport(subWindowWidth, subWindowHeight, 0.0f, 1.0f);
            scissor = Populate::rect2D(static_cast<int32_t>(subWindowWidth),
                                       static_cast<int32_t>(subWindowHeight), 0, 0);
        } else {
            float windowHeight = static_cast<float>(m_Height);
            if (m_guiManager->handles.enableSecondaryView) {
                windowHeight = static_cast<float>(m_Height) / 2;
            }

            float subWindowWidth = static_cast<float>(m_Width) - m_guiManager->handles.info->sidebarWidth;

            viewport = Populate::viewport(subWindowWidth,
                                          windowHeight, 0.0f, 1.0f);

            scissor = Populate::rect2D(subWindowWidth, windowHeight, 0, 0);
        }


        // Define the scissor rectangle for the sub-window
        VkViewport uiViewport = Populate::viewport(static_cast<float>(m_Width), static_cast<float>(m_Height), 0.0f,
                                                   1.0f);

        // Render secondary viewpoints
        vkBeginCommandBuffer(drawCmdBuffers.buffers[currentFrame], &cmdBufInfo);
        drawCmdBuffers.renderPassType = RENDER_PASS_DEPTH_ONLY;

        VkRenderPassBeginInfo depthRenderPassBeginInfo = {};
        depthRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        depthRenderPassBeginInfo.renderPass = depthRenderPass.renderPass; // The second render pass
        depthRenderPassBeginInfo.framebuffer = depthRenderPass.frameBuffers[imageIndex];
        depthRenderPassBeginInfo.renderArea.offset = {0, 0};
        depthRenderPassBeginInfo.renderArea.extent = {m_Width,
                                                      m_Height}; // Set to your off-screen image dimensions
        depthRenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        depthRenderPassBeginInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &depthRenderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.boundRenderPass = depthRenderPassBeginInfo.renderPass;

        VkViewport viewportDepth = Populate::viewport(static_cast<float>(m_Width),
                                                      m_Height, 0.0f, 1.0f);

        VkRect2D scissorDepth = Populate::rect2D(m_Width, m_Height, 0, 0);

        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewportDepth);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissorDepth);


        // Accessing components in a non-copying manner
        for (auto ent: m_registry.view<DepthRenderPassComponent>(entt::exclude<DeleteComponent>)) {
            auto entity = Entity(ent, this);
            if (entity.hasComponent<DefaultGraphicsPipelineComponent2>()) {
                auto &resources = entity.getComponent<DefaultGraphicsPipelineComponent2>();
                resources.draw(&drawCmdBuffers);
            }
        }

        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);

        // Depth pass pipeline barrier before next render pass
        {
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = depthRenderPass.colorImage.image; // TODO type depending on we are using multisample or not
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            // Syncrhonization before main renderpass
            vkCmdPipelineBarrier(
                    drawCmdBuffers.buffers[currentFrame],
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Wait for the render pass to finish
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // Before fragment shader reads
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier
            );

            // Define the depth image layout transition barrier
            barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = depthRenderPass.depthStencil.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                    drawCmdBuffers.buffers[currentFrame],
                    sourceStage, destinationStage,
                    0, // Dependency flags
                    0, nullptr, // No memory barriers
                    0, nullptr, // No buffer barriers
                    1, &barrier // Image barrier
            );
        }

        /// *** Color render pass *** ///
        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = m_Width;
        renderPassBeginInfo.renderArea.extent.height = m_Height;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.framebuffer = frameBuffers[imageIndex];
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.boundRenderPass = renderPassBeginInfo.renderPass;
        drawCmdBuffers.renderPassType = RENDER_PASS_COLOR;
        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);

        // Generate command for copying depth render pass to file
        if (saveDepthPassToFile) {

            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = m_Width * m_Height * 4; // Adjust the size based on your depth format
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;


            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer);
            VkMemoryRequirements memReqs;
            vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);


            VkMemoryAllocateInfo memAllocInfo{};
            memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            memAllocInfo.allocationSize = memReqs.size;
            memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(device, &memAllocInfo, nullptr, &stagingBufferMemory);

            vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = depthRenderPass.depthStencil.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

            vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier
            );


            VkBufferImageCopy copyRegion = {};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent = {m_Width, m_Height, 1};

            vkCmdCopyImageToBuffer(
                    commandBuffer,
                    depthRenderPass.depthStencil.image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    stagingBuffer,
                    1,
                    &copyRegion
            );

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &barrier
            );

            vulkanDevice->flushCommandBuffer(commandBuffer, graphicsQueue);

            // Now copy it to file
            void *data;
            vkMapMemory(device, stagingBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);

            // Here you can use an external library to write the depth data
            // For example, using stb_image_write to write a PNG

            float zNear = 0.1f;
            float zFar = m_cameras[m_selectedCameraTag].m_Zfar;
            float *ptr = reinterpret_cast<float *>(data);
            for (size_t i = 0; i < m_Width * m_Height; ++i) {
                float z_n = 2.0f * ptr[i] - 1.0f; // Back to NDC
                float z_cam = (2.0f * zNear * zFar) / (zFar + zNear - z_n * (zFar - zNear));
                ptr[i] = z_cam;
            }

            if (!std::filesystem::exists(saveFileName.parent_path()))
                std::filesystem::create_directories(saveFileName.parent_path());
            Utils::writeTIFFImage(saveFileName, m_Width, m_Height, reinterpret_cast<float *> (data));
            vkUnmapMemory(device, stagingBufferMemory);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
            vkDestroyBuffer(device, stagingBuffer, nullptr);

            saveDepthPassToFile = false;
        }


        for (auto [entity, skybox, gltfComponent]: m_registry.view<VkRender::SkyboxGraphicsPipelineComponent, GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            skybox.draw(&drawCmdBuffers, currentFrame);
            gltfComponent.model->draw(drawCmdBuffers.buffers[currentFrame]);
        }

        for (auto [entity, resources, gltfComponent]: m_registry.view<VkRender::DefaultPBRGraphicsPipelineComponent, GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            if (!resources.markedForDeletion)
                resources.draw(&drawCmdBuffers, currentFrame, gltfComponent);
            else
                resources.resources[currentFrame].busy = false;
        }


        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<DefaultGraphicsPipelineComponent2>(
                entt::exclude<DeleteComponent, ImageViewComponent, SecondaryRenderViewComponent>)) {
            auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
            resources.draw(&drawCmdBuffers);
        }

        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<CameraGraphicsPipelineComponent>(entt::exclude<DeleteComponent>)) {
            auto &resources = m_registry.get<CameraGraphicsPipelineComponent>(entity);
            resources.draw(&drawCmdBuffers);

        }

        for (auto [entity, resource]: m_registry.view<CustomModelComponent>(entt::exclude<DeleteComponent>).each()) {
            resource.draw(&drawCmdBuffers);
        }


        for (auto entity: m_registry.view<ImageViewComponent>(
                entt::exclude<DeleteComponent>)) {
            auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
            resources.draw(&drawCmdBuffers);
        }

        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);


        if (m_guiManager->handles.enableSecondaryView) {
            VkImageSubresourceRange subresourceRange = {};
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourceRange.levelCount = 1;
            subresourceRange.layerCount = 1;

            Utils::setImageLayout(drawCmdBuffers.buffers[currentFrame], swapchain->images[imageIndex],
                                  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, subresourceRange,
                                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            float subWindowWidth = static_cast<float>(m_Width) - m_guiManager->handles.info->sidebarWidth;
            // Define the viewport
            VkViewport viewport2{};
            viewport2.x = 0.0f;
            viewport2.y = static_cast<float>(m_Height) / 2.0f;  // Start from the middle
            viewport2.width = static_cast<float>(subWindowWidth);
            viewport2.height = static_cast<float>(m_Height) / 2.0f;  // Draw to the bottom
            viewport2.minDepth = 0.0f;
            viewport2.maxDepth = 1.0f;

            // Define the scissor rectangle
            VkRect2D scissor2{};
            scissor2.offset = {0, static_cast<int32_t>(m_Height) / 2};  // Start from the middle
            scissor2.extent = {static_cast<uint32_t>(m_Width - m_guiManager->handles.info->sidebarWidth),
                               m_Height / 2};  // Extend to the bottom

            VkRenderPassBeginInfo renderPassBeginInfoSecondary = Populate::renderPassBeginInfo();
            renderPassBeginInfoSecondary.renderPass = secondRenderPass.renderPass;
            renderPassBeginInfoSecondary.renderArea.offset.x = 0;
            renderPassBeginInfoSecondary.renderArea.offset.y = 0;
            renderPassBeginInfoSecondary.renderArea.extent.width = m_Width;
            renderPassBeginInfoSecondary.renderArea.extent.height = m_Height;
            renderPassBeginInfoSecondary.clearValueCount = clearValues.size();
            renderPassBeginInfoSecondary.pClearValues = clearValues.data();
            renderPassBeginInfoSecondary.framebuffer = frameBuffers[imageIndex];
            vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfoSecondary,
                                 VK_SUBPASS_CONTENTS_INLINE);
            drawCmdBuffers.boundRenderPass = renderPassBeginInfoSecondary.renderPass;
            drawCmdBuffers.renderPassType = RENDER_PASS_SECOND;
            vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport2);
            vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor2);

            for (auto entity: m_registry.view<SecondaryRenderViewComponent>(
                    entt::exclude<DeleteComponent>)) {
                auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
                resources.draw(&drawCmdBuffers);
            }


            vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);
        }


        // Transition color attachment for UI render pass
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = colorImage.image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(
                drawCmdBuffers.buffers[currentFrame],
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
        );


        VkRenderPassBeginInfo uiRenderPassBeginInfo = renderPassBeginInfo;
        uiRenderPassBeginInfo.renderPass = uiRenderPass.renderPass;
        uiRenderPassBeginInfo.clearValueCount = 0;
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &uiRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.boundRenderPass = uiRenderPassBeginInfo.renderPass;
        drawCmdBuffers.renderPassType = RENDER_PASS_UI;


        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &uiViewport);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);
        m_guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame);
        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);

        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers.buffers[currentFrame]))

        */
    }

    bool Renderer::compute() {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(computeCommand.buffers[currentFrame], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording compute command buffer!");
        }
        if (vkEndCommandBuffer(computeCommand.buffers[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record compute command buffer!");
        }
        return true;
    }

    void Renderer::updateUniformBuffers() {
        if (!m_selectedCameraTag.empty()) {
            auto it = m_cameras.find(m_selectedCameraTag);
            if (it != m_cameras.end())
                m_cameras[m_selectedCameraTag].update(frameTimer);
        }
        // update imgui io:
        ImGui::SetCurrentContext(m_mainEditor->guiContext());
        ImGuiIO &mainIO = ImGui::GetIO();
        mainIO.DeltaTime = frameTimer;
        mainIO.WantCaptureMouse = true;
        mainIO.MousePos = ImVec2(mouse.x, mouse.y);
        mainIO.MouseDown[0] = mouse.left;
        mainIO.MouseDown[1] = mouse.right;
        for (auto &editor: m_editors) {
            ImGui::SetCurrentContext(editor->guiContext());
            ImGuiIO &otherIO = ImGui::GetIO();
            otherIO.DeltaTime = frameTimer;
            otherIO.WantCaptureMouse = true;
            otherIO.MousePos = ImVec2(mouse.x - editor->ui().x, mouse.y - editor->ui().y);
            otherIO.MouseDown[0] = mouse.left;
            otherIO.MouseDown[1] = mouse.right;
        }
        updateEditors();
        m_mainEditor->update((frameCounter == 0), frameTimer, &input);
        m_logger->frameNumber = frameID;
        m_sharedContextData.multiSenseRendererBridge->update();
        // TODO reconsider if we should call crl updates here?

        std::string versionRemote;
        /*
        if (m_mainEditor->handles.askUserForNewVersion && m_usageMonitor->getLatestAppVersionRemote(&versionRemote)) {
            std::string localAppVersion = RendererConfig::getInstance().getAppVersion();
            Log::Logger::getInstance()->info("New Version is Available: Local version={}, available version={}",
                                             localAppVersion, versionRemote);
            m_mainEditor->handles.newVersionAvailable = Utils::isLocalVersionLess(localAppVersion, versionRemote);
        }
         */



        //if (keyPress == GLFW_KEY_SPACE) {
        //    m_cameras[m_selectedCameraTag].resetPosition();
        //}
        /**@brief Record commandbuffers for obj models
        // Accessing components in a non-copying manner

        for (auto entity: m_registry.view<DefaultGraphicsPipelineComponent2>()) {
            auto &resources = m_registry.get<DefaultGraphicsPipelineComponent2>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }
        /**@brief Record commandbuffers for obj models
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<CustomModelComponent>()) {
            auto &resources = m_registry.get<CustomModelComponent>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }
        /**@brief Record commandbuffers for obj models
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<CameraGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<CameraGraphicsPipelineComponent>(entity);
            auto &tag = m_registry.get<TagComponent>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        } /**@brief Record commandbuffers for gltf models
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<DefaultPBRGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<DefaultPBRGraphicsPipelineComponent>(entity);
            auto &tag = m_registry.get<TagComponent>(entity);
            const auto &transform = m_registry.get<TransformComponent>(entity);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }

        /*
        // Update GUI
        m_guiManager->handles.info->frameID = frameID;
        m_guiManager->handles.info->applicationRuntime = runTime;
        m_guiManager->update((frameCounter == 0), frameTimer, m_renderUtils.width, m_renderUtils.height, &input);


        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->uiUpdate(&m_guiManager->handles);
        }


        // Load components from UI actions?
        // Load new obj file
        if (m_guiManager->handles.m_paths.updateObjPath) {
            Log::Logger::getInstance()->info("Loading new model from {}",
                                             m_guiManager->handles.m_paths.importFilePath.string());
            std::filesystem::path filename = m_guiManager->handles.m_paths.importFilePath.filename();

            auto entity = createEntity(filename.replace_extension().string());
            auto &component = entity.addComponent<OBJModelComponent>(m_guiManager->handles.m_paths.importFilePath,
                                                                     m_vulkanDevice);

            entity.addComponent<DefaultGraphicsPipelineComponent2>(&m_renderUtils).bind(component);
            entity.addComponent<DepthRenderPassComponent>();
        }
        // Load new gltf file
        if (m_guiManager->handles.m_paths.updateGLTFPath) {
            Log::Logger::getInstance()->info("Loading new model from {}",
                                             m_guiManager->handles.m_paths.importFilePath.string());
            std::filesystem::path filename = m_guiManager->handles.m_paths.importFilePath.filename();
            auto entity = createEntity(filename.replace_extension().string());
            auto &component = entity.addComponent<VkRender::GLTFModelComponent>(
                    m_guiManager->handles.m_paths.importFilePath.string(),
                    m_vulkanDevice);
            auto &sky = findEntityByName("Skybox").getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
            entity.addComponent<VkRender::DefaultPBRGraphicsPipelineComponent>(&m_renderUtils, component, sky);
            entity.addComponent<DepthRenderPassComponent>();

        }

        // Update camera gizmos
        for (auto entity: m_registry.view<CameraGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<CameraGraphicsPipelineComponent>(entity);
            const auto *camera = m_registry.get<CameraComponent>(entity).camera;
            auto &transform = m_registry.get<TransformComponent>(entity);
            transform.setQuaternion(camera->pose.q);
            transform.setPosition(camera->pose.pos);
            const auto &currentCamera = m_cameras[m_selectedCameraTag];
            resources.updateTransform(transform);
            resources.updateView(currentCamera);
            resources.update(currentFrame);
        }
        */
    }

    std::unique_ptr<Editor> Renderer::createEditor(VulkanRenderPassCreateInfo &createInfo) {
        auto editor = createEditorWithUUID(UUID(), createInfo);
        return editor;
    }

    std::unique_ptr<Editor> Renderer::createEditorWithUUID(UUID uuid, VulkanRenderPassCreateInfo &createInfo) {
        return m_editorFactory->createEditor(createInfo.editorTypeDescription, createInfo, uuid);
    }

    void Renderer::recreateEditor(std::unique_ptr<Editor> &editor, VulkanRenderPassCreateInfo &createInfo) {
        auto newEditor = createEditor(createInfo);
        newEditor->ui() = editor->ui();
        editor = std::move(newEditor);
    }

    void Renderer::updateEditors() {
        // Reorder Editors elements according to UI
        for (auto &editor: m_editors) {
            if (editor->ui().changed) {
                // Set a new one
                Log::Logger::getInstance()->info("New Editor requested");
                auto &ci = editor->getCreateInfo();
                ci.editorTypeDescription = editor->ui().selectedType;
                recreateEditor(editor, ci);
            }
        }
        handleEditorResize();
    }

    void Renderer::recordCommands() {
        /** Generate Draw Commands **/
        buildCommandBuffers();
        /** IF WE SHOULD RENDER SECOND IMAGE FOR MOUSE PICKING EVENTS (Reason: let user see PerPixelInformation)
         *  THIS INCLUDES RENDERING SELECTED OBJECTS AND COPYING CONTENTS BACK TO CPU INSTEAD OF DISPLAYING TO SCREEN **/
    }

    void Renderer::windowResized(int32_t dx, int32_t dy, double widthScale, double heightScale) {

        if ((m_width > 0.0) && (m_height > 0.0)) {
            for (auto &camera: m_cameras)
                camera.second.updateAspectRatio(static_cast<float>(m_width) / static_cast<float>(m_height));
        }
        Widgets::clear();
        // Update gui with new res
        //m_guiManager->update((frameCounter == 0), frameTimer, m_renderUtils.width, m_renderUtils.height, &input);

        // Notify scripts
        auto view = m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            //script.script->windowResize(&m_guiManager->handles);
        }
        // Destroy framebuffer
        for (auto &fb: m_frameBuffers) {
            vkDestroyFramebuffer(device, fb, nullptr);
        }

        vkDestroyImageView(device, m_depthStencil.view, nullptr);
        vmaDestroyImage(m_allocator, m_depthStencil.image, m_depthStencil.allocation);

        vkDestroyImageView(device, m_colorImage.view, nullptr);
        vmaDestroyImage(m_allocator, m_colorImage.image, m_colorImage.allocation);

        createColorResources();
        createDepthStencil();
        createMainRenderPass();


        if (dx != 0)
            Editor::windowResizeEditorsHorizontal(dx, widthScale, m_editors, m_width);
        if (dy != 0)
            Editor::windowResizeEditorsVertical(dy, heightScale, m_editors, m_height);

        for (auto &editor: m_editors) {
            auto &ci = editor->getCreateInfo();
            ci.appHeight = m_height;
            ci.appWidth = m_width;
            editor->resize(ci);
        }
        auto &ci = m_mainEditor->getCreateInfo();
        ci.width = m_width;
        ci.height = m_height;
        ci.appWidth = m_width;
        ci.appHeight = m_height;
        ci.frameBuffers = m_frameBuffers.data();
        m_mainEditor->resize(ci);
    }

    void Renderer::cleanUp() {
        if (std::filesystem::exists(Utils::getRuntimeConfigFilePath())) {
            std::filesystem::remove(Utils::getRuntimeConfigFilePath());
            Log::Logger::getInstance()->info("Removed runtime config file before cleanup {}",
                                             Utils::getRuntimeConfigFilePath().string().c_str());
        }

        auto startTime = std::chrono::steady_clock::now();

        m_usageMonitor->userEndSession();
        RendererConfig::getInstance().getUserSetting().applicationWidth = m_width;
        RendererConfig::getInstance().getUserSetting().applicationHeight = m_height;

        if (m_usageMonitor->hasUserLogCollectionConsent() &&
            RendererConfig::getInstance().getUserSetting().sendUsageLogOnExit)
            m_usageMonitor->sendUsageLog();

        RendererConfig::getInstance().saveSettings(this);
        auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Sending logs on exit took {}s", timeSpan.count());

        startTime = std::chrono::steady_clock::now();
        // Shutdown GUI manually since it contains thread. Not strictly necessary but nice to have
        //m_guiManager.reset();
        timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Deleting GUI on exit took {}s", timeSpan.count());

        startTime = std::chrono::steady_clock::now();

        auto view = m_registry.view<entt::any>();
        // Step 3: Destroy entities
        for (auto entity: view) {
            m_registry.destroy(entity);
        }
        timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Deleting entities on exit took {}s", timeSpan.count());
        // Destroy framebuffer
        for (auto &fb: m_frameBuffers) {
            vkDestroyFramebuffer(device, fb, nullptr);
        }

        m_editors.clear();
        m_mainEditor.reset();
    }

    void Renderer::handleEditorResize() {
        //// UPDATE EDITOR WITH UI EVENTS - Very little logic here
        for (auto &editor: m_editors) {
            Editor::handleHoverState(editor, mouse);
            Editor::handleClickState(editor, mouse);
        }

        bool showHandCursor = false;
        bool showCrosshairCursor = false;
        bool anyCornerClicked = false;
        bool anyResizeHovered = false;
        bool horizontalResizeHovered = false;
        for (auto &editor: m_editors) {
            if (editor->ui().cornerBottomLeftHovered && editor->getCreateInfo().resizeable) showHandCursor = true;
            if (editor->ui().cornerBottomLeftClicked && editor->getCreateInfo().resizeable) showCrosshairCursor = true;

            if (editor->ui().cornerBottomLeftClicked) anyCornerClicked = true;
            if (editor->ui().resizeHovered) anyResizeHovered = true;
            if (EditorBorderState::Left == editor->ui().lastHoveredBorderType ||
                EditorBorderState::Right == editor->ui().lastHoveredBorderType)
                horizontalResizeHovered = true;
        }

        //// UPDATE EDITOR Based on UI EVENTS
        for (auto &editor: m_editors) {
            // Dont update the editor if managed by another instance
            if (!editor->ui().indirectlyActivated) {
                Editor::handleIndirectClickState(m_editors, editor, mouse);
            }
            Editor::handleDragState(editor, mouse);
        }

        Editor::checkIfEditorsShouldMerge(m_editors);

        resizeEditors(anyCornerClicked);
        for (auto &editor: m_editors) {
            editor->update((frameCounter == 0), frameTimer, &input);
            if (!mouse.left) {
                if (editor->ui().indirectlyActivated) {
                    Editor::handleHoverState(editor, mouse);
                    editor->ui().lastClickedBorderType = None;
                    editor->ui().active = false;
                    editor->ui().indirectlyActivated = false;
                }
                editor->ui().resizeActive = false;
                editor->ui().cornerBottomLeftClicked = false;
                editor->ui().dragHorizontal = false;
                editor->ui().dragVertical = false;
                editor->ui().dragActive = false;
                editor->ui().splitting = false;
                editor->ui().lastPressedPos = glm::ivec2(-1, -1);
                editor->ui().dragDelta = glm::ivec2(0, 0);
                editor->ui().cursorDelta = glm::ivec2(0, 0);
            }
            if (!mouse.right) {
                editor->ui().rightClickBorder = false;
                editor->ui().lastRightClickedBorderType = None;
            }
            if (mouse.left && mouse.action == GLFW_PRESS) {
                Log::Logger::getInstance()->info("We Left-clicked Editor: {}'s area :{}",
                                                 editor->ui().index,
                                                 editor->ui().lastClickedBorderType);
            }

            if (mouse.right && mouse.action == GLFW_PRESS) {
                Log::Logger::getInstance()->info("We Right-clicked Editor: {}'s area :{}. Merge? {}",
                                                 editor->ui().index,
                                                 editor->ui().lastClickedBorderType,
                                                 editor->ui().rightClickBorder);
            }
        }
        bool splitEditors = false;
        uint32_t splitEditorIndex = UINT32_MAX;
        for (size_t index = 0; auto &editor: m_editors) {
            if (editor->getCreateInfo().resizeable &&
                editor->ui().cornerBottomLeftClicked &&
                editor->ui().width > 100 && editor->ui().height > 100 &&
                (editor->ui().dragHorizontal || editor->ui().dragVertical) &&
                !editor->ui().splitting) {
                splitEditors = true;
                splitEditorIndex = index;
            }
            index++;
        }
        if (splitEditors) {
            splitEditor(splitEditorIndex);
        }

        bool mergeEditor = false;
        std::array<UUID, 2> editorsUUID;
        for (size_t index = 0; auto &editor: m_editors) {
            if (editor->ui().shouldMerge) {
                editorsUUID[index] = editor->getUUID();
                index++;
                mergeEditor = true;
            }
        }
        if (mergeEditor)
            mergeEditors(editorsUUID);
        //
        if (showCrosshairCursor) {
            glfwSetCursor(window, m_cursors.crossHair);
        } else if (showHandCursor) {
            glfwSetCursor(window, m_cursors.hand);
        } else if (anyResizeHovered) {
            glfwSetCursor(window, horizontalResizeHovered ? m_cursors.resizeHorizontal : m_cursors.resizeVertical);
        } else {
            glfwSetCursor(window, m_cursors.arrow);
        }
    }


    void Renderer::resizeEditors(bool anyCornerClicked) {
        bool isValidResizeAll = true;
        for (auto &editor: m_editors) {
            if (editor->ui().resizeActive && (!anyCornerClicked || editor->ui().splitting)) {
                auto createInfo = getNewEditorCreateInfo(editor);
                if (!Editor::isValidResize(createInfo, editor))
                    isValidResizeAll = false;
            }
        }
        if (isValidResizeAll) {
            for (auto &editor: m_editors) {
                if (editor->ui().resizeActive && (!anyCornerClicked || editor->ui().splitting)) {
                    auto createInfo = getNewEditorCreateInfo(editor);
                    editor->resize(createInfo);
                }
            }
        }
    }

    void Renderer::splitEditor(uint32_t splitEditorIndex) {
        auto &editor = m_editors[splitEditorIndex];
        VulkanRenderPassCreateInfo &editorCreateInfo = editor->getCreateInfo();
        VulkanRenderPassCreateInfo newEditorCreateInfo(m_frameBuffers.data(), m_guiResources, this,
                                                       &m_sharedContextData);
        VulkanRenderPassCreateInfo::copy(&newEditorCreateInfo, &editorCreateInfo);

        if (editor->ui().dragHorizontal) {
            editorCreateInfo.width -= editor->ui().dragDelta.x;
            editorCreateInfo.x += editor->ui().dragDelta.x;
            newEditorCreateInfo.width = editor->ui().dragDelta.x;
        } else {
            editorCreateInfo.height += editor->ui().dragDelta.y;
            newEditorCreateInfo.height = -editor->ui().dragDelta.y;
            newEditorCreateInfo.y = editorCreateInfo.height + editorCreateInfo.y;
        }
        newEditorCreateInfo.editorIndex = m_editors.size();
        if (!editor->validateEditorSize(newEditorCreateInfo) ||
            !editor->validateEditorSize(editorCreateInfo))
            return;
        editor->resize(editorCreateInfo);
        auto newEditor = createEditor(newEditorCreateInfo);
        editor->ui().resizeActive = true;
        newEditor->ui().resizeActive = true;
        editor->ui().active = true;
        newEditor->ui().active = true;
        editor->ui().splitting = true;
        newEditor->ui().splitting = true;
        if (editor->ui().dragHorizontal) {
            editor->ui().lastClickedBorderType = EditorBorderState::Left;
            newEditor->ui().lastClickedBorderType = EditorBorderState::Right;
        } else {
            editor->ui().lastClickedBorderType = EditorBorderState::Bottom;
            newEditor->ui().lastClickedBorderType = EditorBorderState::Top;
        }
        m_editors.push_back(std::move(newEditor));
    }

    void Renderer::mergeEditors(const std::array<UUID, 2> &mergeEditorIndices) {
        UUID id1 = mergeEditorIndices[0];
        UUID id2 = mergeEditorIndices[1];

        auto &editor1 = findEditorByUUID(id1);
        auto &editor2 = findEditorByUUID(id2);

        if (!editor1 || !editor2) {
            Log::Logger::getInstance()->info("Wanted to merge editors: {} and {} but they were not found",
                                             id1.operator std::string(), id2.operator std::string());
            return;
        }
        // Implement your merging logic here
        // For example, combine editor2's properties into editor1
        // editor1.someProperty += editor2.someProperty;
        editor1->ui().shouldMerge = false;
        editor2->ui().shouldMerge = false;

        auto &ci1 = editor1->getCreateInfo();
        auto &ci2 = editor2->getCreateInfo();

        int32_t newX = std::min(ci1.x, ci2.x);
        int32_t newY = std::min(ci1.y, ci2.y);
        int32_t newWidth = ci1.width + ci2.width;
        int32_t newHeight = ci1.height + ci2.height;

        if (editor1->ui().lastRightClickedBorderType & EditorBorderState::HorizontalBorders) {
            ci1.height = newHeight;
        } else if (editor1->ui().lastRightClickedBorderType & EditorBorderState::VerticalBorders) {
            ci1.width = newWidth;
        }
        ci1.x = newX;
        ci1.y = newY;

        Log::Logger::getInstance()->info("Merging editor {} into editor {}.", editor2->ui().index, editor1->ui().index);

        auto editor2UUID = editor2->getUUID();
        // Remove editor2 safely based on UUID
        m_editors.erase(
                std::remove_if(m_editors.begin(), m_editors.end(),
                               [editor2UUID](const std::unique_ptr<Editor> &editor) {
                                   return editor->getUUID() == editor2UUID;
                               }),
                m_editors.end()
        );
        editor1->resize(ci1);
    }

    VulkanRenderPassCreateInfo Renderer::getNewEditorCreateInfo(std::unique_ptr<Editor> &editor) {
        VulkanRenderPassCreateInfo newEditorCI(m_frameBuffers.data(), m_guiResources, this, &m_sharedContextData);
        VulkanRenderPassCreateInfo::copy(&newEditorCI, &editor->getCreateInfo());

        switch (editor->ui().lastClickedBorderType) {
            case EditorBorderState::Left:
                newEditorCI.x = editor->ui().x + editor->ui().cursorDelta.x;
                newEditorCI.width = editor->ui().width - editor->ui().cursorDelta.x;
                break;
            case EditorBorderState::Right:
                newEditorCI.width = editor->ui().width + editor->ui().cursorDelta.x;
                break;
            case EditorBorderState::Top:
                newEditorCI.y = editor->ui().y + editor->ui().cursorDelta.y;
                newEditorCI.height = editor->ui().height - editor->ui().cursorDelta.y;
                break;
            case EditorBorderState::Bottom:
                newEditorCI.height = editor->ui().height + editor->ui().cursorDelta.y;
                break;
            default:
                Log::Logger::getInstance()->trace(
                        "Resize is somehow active but we have not clicked any borders: {}",
                        editor->ui().index);
                break;
        }
        return newEditorCI;
    }

    void Renderer::mouseMoved(float x, float y, bool &handled) {
        mouse.insideApp = !(x < 0 || x > m_width || y < 0 || y > m_height);
        float dx = x - mouse.x;
        float dy = y - mouse.y;
        mouse.dx += dx;
        mouse.dy += dy;
        Log::Logger::getInstance()->trace("Cursor velocity: ({},{}), pos: ({},{})", mouse.dx, mouse.dy, mouse.x,
                                          mouse.y);
        // UPdate camera if we have one selected
        if (!m_selectedCameraTag.empty()) {
            auto it = m_cameras.find(m_selectedCameraTag);
            if (it != m_cameras.end()) {
                if (mouse.left) {
                    // && !mouseButtons.middle) {
                    m_cameras[m_selectedCameraTag].rotate(dx, dy);
                }
                //if (mouseButtons.left && m_guiManager->handles.renderer3D)
                //    m_cameras[m_selectedCameraTag].rotate(dx, dy);
                if (mouse.right) {
                    if (m_cameras[m_selectedCameraTag].m_type == Camera::arcball)
                        m_cameras[m_selectedCameraTag].translate(glm::vec3(-dx * 0.005f, -dy * 0.005f, 0.0f));
                    else
                        m_cameras[m_selectedCameraTag].translate(-dx * 0.01f, -dy * 0.01f);
                }
                if (mouse.middle && m_cameras[m_selectedCameraTag].m_type == Camera::flycam) {
                    m_cameras[m_selectedCameraTag].translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
                } else if (mouse.middle && m_cameras[m_selectedCameraTag].m_type == Camera::arcball) {
                    //camera.orbitPan(static_cast<float>() -dx * 0.01f, static_cast<float>() -dy * 0.01f);
                }
            }
        }
        mouse.x = x;
        mouse.y = y;
        handled = true;
    }

    void Renderer::mouseScroll(float change) {
        /*
        if (m_guiManager->handles.renderer3D) {
            m_cameras[m_selectedCameraTag].setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        }
         */
    }

    Entity Renderer::createEntity(const std::string &name) {
        return createEntityWithUUID(UUID(), name);
    }


    Entity Renderer::createEntityWithUUID(UUID uuid, const std::string &name) {
        Entity entity = {m_registry.create(), this};
        entity.addComponent<IDComponent>(uuid);
        entity.addComponent<TransformComponent>();
        auto &tag = entity.addComponent<TagComponent>();
        tag.Tag = name.empty() ? "Entity" : name;
        Log::Logger::getInstance()->info("Created Entity with UUID: {} and Tag: {}",
                                         entity.getUUID().operator std::string(), entity.getName());
        m_entityMap[uuid] = entity;

        return entity;
    }

    Entity Renderer::findEntityByName(std::string_view name) {
        auto view = m_registry.view<TagComponent>();
        for (auto entity: view) {
            const TagComponent &tc = view.get<TagComponent>(entity);
            if (tc.Tag == name)
                return Entity{entity, this};
        }
        return {};
    }

    std::unique_ptr<Editor> &Renderer::findEditorByUUID(const UUID &uuid) {
        for (auto &editor: m_editors) {
            if (uuid == editor->getUUID()) {
                return editor;
            }
        }
    }

    void Renderer::postRenderActions() {
        // Reset mousewheel across imgui contexts
        /*
        for (std::vector<ImGuiContext *> list = {m_mainEditor->m_guiManager->m_imguiContext,
                                                 m_editors[0].m_guiManager->m_imguiContext}; auto &ctx : list) {
            ImGui::SetCurrentContext(ctx);
            ImGuiIO &io = ImGui::GetIO();
            io.MouseWheel = 0;
        }
        */
    }

    void Renderer::destroyEntity(Entity entity) {
        if (!entity) {
            Log::Logger::getInstance()->warning("Attempted to delete an entity that doesn't exist");
            return;
        }
        // Checking if the entity is still valid before attempting to delete
        if (m_registry.valid(entity)) {
            Log::Logger::getInstance()->info("Deleting Entity with UUID: {} and Tag: {}",
                                             entity.getUUID().operator std::string(), entity.getName());

            // Perform the deletion
            m_entityMap.erase(entity.getUUID());
            m_registry.destroy(entity);
        } else {
            Log::Logger::getInstance()->warning(
                    "Attempted to delete an invalid or already deleted entity with UUID: {}",
                    entity.getUUID().operator std::string());
        }
    }

    Camera &Renderer::createNewCamera(const std::string &name, uint32_t width, uint32_t height) {
        auto e = createEntity(name);
        auto camera = Camera(m_width, m_height);
        m_cameras[name] = camera;
        auto &c = e.addComponent<CameraComponent>(&m_cameras[m_selectedCameraTag]);
        //auto &gizmo = e.addComponent<CameraGraphicsPipelineComponent>(&m_renderUtils);
        auto &transform = e.getComponent<TransformComponent>();
        transform.scale = glm::vec3(0.2f, 0.2f, 0.2f);

        return m_cameras[name];
    }

    Camera &Renderer::getCamera() {
        if (!m_selectedCameraTag.empty()) {
            auto it = m_cameras.find(m_selectedCameraTag);
            if (it != m_cameras.end()) {
                return m_cameras[m_selectedCameraTag];
            }
        } // TODO create a new camera with tag if it doesn't exist
    }

    Camera &Renderer::getCamera(std::string tag) {
        if (!m_selectedCameraTag.empty()) {
            auto it = m_cameras.find(tag);
            if (it != m_cameras.end()) {
                return m_cameras[tag];
            }
        }
        // TODO create a new camera with tag if it doesn't exist
    }

    void Renderer::keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
        m_cameras[m_selectedCameraTag].keys.up = input.keys.up;
        m_cameras[m_selectedCameraTag].keys.down = input.keys.down;
        m_cameras[m_selectedCameraTag].keys.left = input.keys.left;
        m_cameras[m_selectedCameraTag].keys.right = input.keys.right;
    }

    void Renderer::createDepthStencil() {
        std::string description = "Renderer:";
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = depthFormat;
        imageCI.extent = {m_width, m_height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = msaaSamples;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkResult result = vmaCreateImage(allocator(), &imageCI, &allocInfo, &m_depthStencil.image,
                                         &m_depthStencil.allocation, nullptr);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image");
        vmaSetAllocationName(allocator(), m_depthStencil.allocation, (description + "DepthStencil").c_str());
        VALIDATION_DEBUG_NAME(m_vulkanDevice->m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_depthStencil.image), VK_OBJECT_TYPE_IMAGE,
                              (description + "DepthImage").c_str());

        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = m_depthStencil.image;
        imageViewCI.format = depthFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(m_vulkanDevice->m_LogicalDevice, &imageViewCI, nullptr,
                                   &m_depthStencil.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image view");

        VALIDATION_DEBUG_NAME(m_vulkanDevice->m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_depthStencil.view), VK_OBJECT_TYPE_IMAGE_VIEW,
                              (description + "DepthView").c_str());

        VkCommandBuffer copyCmd = m_vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 1;

        Utils::setImageLayout(copyCmd, m_depthStencil.image, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, subresourceRange,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        m_vulkanDevice->flushCommandBuffer(copyCmd, graphicsQueue, true);
    }


    void Renderer::createColorResources() {
        std::string description = "Renderer:";

        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = swapchain->colorFormat;
        imageCI.extent = {m_width, m_height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = msaaSamples;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkResult result = vmaCreateImage(allocator(), &imageCI, &allocInfo, &m_colorImage.image,
                                         &m_colorImage.allocation, nullptr);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create color image");
        VALIDATION_DEBUG_NAME(m_vulkanDevice->m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_colorImage.image), VK_OBJECT_TYPE_IMAGE,
                              (description + "ColorImageResource").c_str());
        // Set user data for debugging
        //vmaSetAllocationUserData(allocator(), m_colorImage.allocation, (void*)((description + "ColorResource").c_str()));

        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = m_colorImage.image;
        imageViewCI.format = swapchain->colorFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        result = vkCreateImageView(m_vulkanDevice->m_LogicalDevice, &imageViewCI, nullptr,
                                   &m_colorImage.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create color image view");
        VALIDATION_DEBUG_NAME(m_vulkanDevice->m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_colorImage.view), VK_OBJECT_TYPE_IMAGE_VIEW,
                              (description + "ColorViewResource").c_str());
    }


    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

    template<typename T>
    void Renderer::onComponentAdded(Entity entity, T &component) {
        static_assert(sizeof(T) == 0);
    }

    template<>
    void Renderer::onComponentAdded<IDComponent>(Entity entity, IDComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<TransformComponent>(Entity entity, TransformComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<CameraComponent>(Entity entity, CameraComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<ScriptComponent>(Entity entity, ScriptComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<TagComponent>(Entity entity, TagComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<Rigidbody2DComponent>(Entity entity, Rigidbody2DComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<TextComponent>(Entity entity, TextComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<OBJModelComponent>(Entity entity, OBJModelComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<DefaultGraphicsPipelineComponent>(Entity entity, DefaultGraphicsPipelineComponent &component) {
    }

    DISABLE_WARNING_POP
};
