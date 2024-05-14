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


#include <array>
#include <stb_image_write.h>

#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"

#include "Viewer/Renderer/Components.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Core/UUID.h"

#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/DefaultPBRGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/SecondaryCameraComponent.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"
#include "Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"

namespace VkRender {

    Renderer::Renderer(const std::string &title) : VulkanRenderer(title) {
        VkRender::RendererConfig &config = VkRender::RendererConfig::getInstance();
        this->m_Title = title;
        // Create Log C++ Interface
        Log::Logger::getInstance()->setLogLevel(config.getLogLevel());
        pLogger = Log::Logger::getInstance();


        VulkanRenderer::initVulkan();
        VulkanRenderer::prepare();
        backendInitialized = true;
        // Create default camera object
        m_Camera = Camera(m_Width, m_Height);
        auto e = createEntity("Default");
        auto &c = e.addComponent<CameraComponent>(m_Camera);
        cameras[selectedCameraTag] = &c.camera;

        pLogger->info("Initialized Backend");
        config.setGpuDevice(physicalDevice);

        renderData.height = m_Height;
        renderData.width = m_Width;
        // Start up usage monitor
        usageMonitor = std::make_shared<UsageMonitor>();
        usageMonitor->loadSettingsFromFile();
        usageMonitor->userStartSession(rendererStartTime);

        guiManager = std::make_unique<VkRender::GuiManager>(vulkanDevice, renderPass, m_Width, m_Height,
                                                            msaaSamples,
                                                            swapchain->imageCount);
        guiManager->handles.mouse = &mouseButtons;
        guiManager->handles.usageMonitor = usageMonitor;
        guiManager->handles.m_cameraSelection.info[selectedCameraTag].type = cameras[selectedCameraTag]->m_type;

        guiManager->handles.m_context = this;
        prepareRenderer();
        pLogger->info("Prepared Renderer");
    }


    void Renderer::prepareRenderer() {
        cameras[selectedCameraTag]->setType(VkRender::Camera::arcball);
        cameras[selectedCameraTag]->setPerspective(60.0f, static_cast<float>(m_Width) / static_cast<float>(m_Height));
        cameras[selectedCameraTag]->resetPosition();
        cameraConnection = std::make_unique<VkRender::MultiSense::CameraConnection>();

        // TODO Make dynamic
        VulkanRenderer::setupSecondaryRenderPasses(false);
        renderUtils.secondaryRenderPasses = &secondaryRenderPasses;

        // Run Once
        renderUtils.device = vulkanDevice;
        renderUtils.instance = &instance;
        renderUtils.renderPass = &renderPass;
        renderUtils.msaaSamples = msaaSamples;
        renderUtils.swapchainImages = swapchain->imageCount;
        renderUtils.queueSubmitMutex = &queueSubmitMutex;
        renderUtils.fence = &waitFences;
        renderUtils.swapchainIndex = currentFrame;


        std::ifstream infile(Utils::getAssetsPath().append("Generated/Scripts.txt").string());
        std::string line;
        while (std::getline(infile, line)) {
            // Skip comment # line
            if (line.find('#') != std::string::npos || line.find("Skybox") != std::string::npos)
                continue;
            availableScriptNames.emplace_back(line);
        }

        for (const auto &scriptName: availableScriptNames) {
            auto e = createEntity(scriptName);
            e.addComponent<ScriptComponent>(e.getName(), this);
        }

        auto view = m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->setup();
        }
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

    // Helper function to attempt cleanup and destroy the entity if successful
    template<typename T>
    bool Renderer::tryCleanupAndDestroy(Entity& entity, int currentFrame) {
        if (entity.hasComponent<T>()) {
            auto& res = entity.getComponent<T>();
            if (res.cleanUp(currentFrame)) {
                destroyEntity(Entity(entity, this));
                return true; // Entity was deleted from register
            }
            return true; // We found the component and notified we want to delete vulkan resources -- Try again entity not deleted from register
        }
        return false; // No render component -- just delete entity
    }


    void Renderer::processDeletions() {
        // TODO Specify cleanup routine for each component individually. Let the component manage the deletion itself
        // Check for PBR elements and if we should delay deletion
        for (auto [entity, gltfModel, deleteComponent]: m_registry.view<RenderResource::DefaultPBRGraphicsPipelineComponent, DeleteComponent>().each()) {
            gltfModel.markedForDeletion = true;
            bool readyForDeletion = true;
            for (const auto &resource: gltfModel.resources) {
                if (resource.busy)
                    readyForDeletion = false;
            }
            if (readyForDeletion) {
                destroyEntity(Entity(entity, this));
            }
        }

        // Delete Entities:
        for (auto [entity_id, deleteComponent]: m_registry.view<DeleteComponent>().each()) {
            Entity entity(entity_id, this);

            // Simplify the cleanup check using a templated helper function
            if (tryCleanupAndDestroy<DefaultGraphicsPipelineComponent2>(entity, currentFrame) ||
                tryCleanupAndDestroy<CameraGraphicsPipelineComponent>(entity, currentFrame) ||
                tryCleanupAndDestroy<CustomModelComponent>(entity, currentFrame)) {
                continue;
            }

            // If no specific components were found or needed cleanup, destroy the entity
            destroyEntity(entity);

        }
    }


    void Renderer::buildCommandBuffers() {
        processDeletions();
        /**@brief Record command buffers for skybox */
        VkCommandBufferBeginInfo cmdBufInfo = Populate::commandBufferBeginInfo();
        cmdBufInfo.flags = 0;
        cmdBufInfo.pInheritanceInfo = nullptr;
        std::array<VkClearValue, 3> clearValues{};
        if (guiManager->handles.renderer3D) {
            clearValues[0] = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
            clearValues[2] = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        } else {
            clearValues[0] = {{{guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                                guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]}}};
            clearValues[2] = {{{guiManager->handles.clearColor[0], guiManager->handles.clearColor[1],
                                guiManager->handles.clearColor[2], guiManager->handles.clearColor[3]}}};
        }
        clearValues[1].depthStencil = {1.0f, 0};

        const VkViewport viewport = Populate::viewport(static_cast<float>(m_Width), static_cast<float>(m_Height), 0.0f, 1.0f);
        const VkRect2D scissor = Populate::rect2D(static_cast<int32_t>(m_Width), static_cast<int32_t>(m_Height), 0, 0);

        VkViewport uiViewport = Populate::viewport(static_cast<float>(m_Width), static_cast<float>(m_Height), 0.0f, 1.0f);
        //VkViewport sceneViewport = Populate::viewport(static_cast<float>(subWindowWidth), static_cast<float>(subWindowHeight), 0.0f, 1.0f);

        // Render secondary viewpoints
        vkBeginCommandBuffer(drawCmdBuffers.buffers[currentFrame], &cmdBufInfo);

        drawCmdBuffers.renderPassType = RENDER_PASS_DEPTH_ONLY;
        for (const auto &render: secondaryRenderPasses) {
            VkRenderPassBeginInfo secondaryRenderPassBeginInfo = {};
            secondaryRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            secondaryRenderPassBeginInfo.renderPass = render.renderPass; // The second render pass
            secondaryRenderPassBeginInfo.framebuffer = render.frameBuffers[imageIndex];
            secondaryRenderPassBeginInfo.renderArea.offset = {0, 0};
            secondaryRenderPassBeginInfo.renderArea.extent = {m_Width,
                                                              m_Height}; // Set to your off-screen image dimensions
            secondaryRenderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            secondaryRenderPassBeginInfo.pClearValues = clearValues.data();

            vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &secondaryRenderPassBeginInfo,
                                 VK_SUBPASS_CONTENTS_INLINE);
            drawCmdBuffers.boundRenderPass = secondaryRenderPassBeginInfo.renderPass;
            vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
            vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);


            /**@brief Record commandbuffers for obj models */
            // Accessing components in a non-copying manner
            for (auto entity: m_registry.view<VkRender::SecondaryRenderPassComponent>(entt::exclude<DeleteComponent>)) {
                auto &resources = m_registry.get<VkRender::DefaultGraphicsPipelineComponent2>(entity);
                resources.draw(&drawCmdBuffers);
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
                barrier.image = render.colorImage.image; // TODO type depending on we are using multisample or not
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
                barrier.image = secondaryRenderPasses[0].depthStencil.image;
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

        }
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
            auto &secondaryRenderPass = secondaryRenderPasses.front();

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
            barrier.image = secondaryRenderPass.depthStencil.image;
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
                    secondaryRenderPass.depthStencil.image,
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
            float zFar = cameras[selectedCameraTag]->m_Zfar;
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

        /**@brief Record command buffers for skybox */
        for (auto [entity, skybox, gltfComponent]: m_registry.view<RenderResource::SkyboxGraphicsPipelineComponent, VkRender::GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            skybox.draw(&drawCmdBuffers, currentFrame);
            gltfComponent.model->draw(drawCmdBuffers.buffers[currentFrame]);
        }

        /**@brief Record commandbuffers for gltf models */
        for (auto [entity, resources, gltfComponent]: m_registry.view<RenderResource::DefaultPBRGraphicsPipelineComponent, VkRender::GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            if (!resources.markedForDeletion)
                resources.draw(&drawCmdBuffers, currentFrame, gltfComponent);
            else
                resources.resources[currentFrame].busy = false;
        }


        /**@brief Record commandbuffers for obj models */
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<VkRender::DefaultGraphicsPipelineComponent2>(
                entt::exclude<DeleteComponent, ImageViewComponent>)) {
            auto &resources = m_registry.get<VkRender::DefaultGraphicsPipelineComponent2>(entity);
            resources.draw(&drawCmdBuffers);
        }

        /**@brief Record commandbuffers for custom camera models */
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<VkRender::CameraGraphicsPipelineComponent>(entt::exclude<DeleteComponent>)) {
            auto &resources = m_registry.get<VkRender::CameraGraphicsPipelineComponent>(entity);
            resources.draw(&drawCmdBuffers);

        }

        /**@brief Record commandbuffers for Custom models (GRID) */
        for (auto [entity, resource]: m_registry.view<CustomModelComponent>(entt::exclude<DeleteComponent>).each()) {
            resource.draw(&drawCmdBuffers);
        }
        /** Generate UI draw commands **/
        guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame);

        /**@brief Record commandbuffers for obj models */
        for (auto entity: m_registry.view<VkRender::ImageViewComponent>(
                entt::exclude<DeleteComponent>)) {
            auto &resources = m_registry.get<VkRender::DefaultGraphicsPipelineComponent2>(entity);
            resources.draw(&drawCmdBuffers);
        }

        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);

        CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers.buffers[currentFrame]))
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
        if (!selectedCameraTag.empty()) {
            auto it = cameras.find(selectedCameraTag);
            if (it != cameras.end())
                cameras[selectedCameraTag]->update(frameTimer);
        }

        selectedCameraTag = guiManager->handles.m_cameraSelection.tag;
        renderData.camera = cameras[selectedCameraTag];
        renderData.deltaT = frameTimer;
        renderData.frameID = frameID;
        renderData.height = m_Height;
        renderData.width = m_Width;
        renderData.crlCamera = &cameraConnection->camPtr;
        renderUtils.swapchainIndex = currentFrame;
        renderUtils.input = &input;
        // New version available?
        std::string versionRemote;
        if (guiManager->handles.askUserForNewVersion && usageMonitor->getLatestAppVersionRemote(&versionRemote)) {
            std::string localAppVersion = VkRender::RendererConfig::getInstance().getAppVersion();
            Log::Logger::getInstance()->info("New Version is Available: Local version={}, available version={}",
                                             localAppVersion, versionRemote);
            guiManager->handles.newVersionAvailable = Utils::isLocalVersionLess(localAppVersion, versionRemote);
        }
        pLogger->frameNumber = frameID;
        if (keyPress == GLFW_KEY_SPACE) {
            cameras[selectedCameraTag]->resetPosition();
        }


        auto view = m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->update();
        }

        /**@brief Record commandbuffers for obj models */
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<VkRender::DefaultGraphicsPipelineComponent2>()) {
            auto &resources = m_registry.get<VkRender::DefaultGraphicsPipelineComponent2>(entity);
            resources.update(currentFrame);
        }
        /**@brief Record commandbuffers for obj models */
        // Accessing components in a non-copying manner
        for (auto entity: m_registry.view<VkRender::CameraGraphicsPipelineComponent>()) {
            auto &resources = m_registry.get<VkRender::CameraGraphicsPipelineComponent>(entity);
            resources.update(currentFrame);
        }

        // Update GUI
        guiManager->handles.info->frameID = frameID;
        guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->uiUpdate(&guiManager->handles);
        }
    }


    void Renderer::recordCommands() {
        /** Generate Draw Commands **/
        buildCommandBuffers();
        /** IF WE SHOULD RENDER SECOND IMAGE FOR MOUSE PICKING EVENTS (Reason: let user see PerPixelInformation)
         *  THIS INCLUDES RENDERING SELECTED OBJECTS AND COPYING CONTENTS BACK TO CPU INSTEAD OF DISPLAYING TO SCREEN **/

    }

    void Renderer::windowResized() {
        renderData.camera = cameras[selectedCameraTag];
        renderData.deltaT = frameTimer;
        renderData.frameID = frameID;
        renderData.height = m_Height;
        renderData.width = m_Width;
        renderData.crlCamera = &cameraConnection->camPtr;
        renderData.renderPassIndex = currentFrame;

        if ((m_Width > 0.0) && (m_Height > 0.0)) {
            for (auto &camera: cameras)
                camera.second->updateAspectRatio(static_cast<float>(m_Width) / static_cast<float>(m_Height));
        }
        Widgets::clear();
        // Update gui with new res
        guiManager->update((frameCounter == 0), frameTimer, renderData.width, renderData.height, &input);

        // Notify scripts
        auto view = m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->windowResize(&guiManager->handles);
        }

    }


    void Renderer::cleanUp() {
        auto startTime = std::chrono::steady_clock::now();

        usageMonitor->userEndSession();

        if (usageMonitor->hasUserLogCollectionConsent() &&
            VkRender::RendererConfig::getInstance().getUserSetting().sendUsageLogOnExit)
            usageMonitor->sendUsageLog();

        auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Sending logs on exit took {}s", timeSpan.count());

        for (auto &dev: guiManager->handles.devices) {
            dev.interruptConnection = true; // Disable all current connections if user wants to exit early
            cameraConnection->saveProfileAndDisconnect(&dev);
        }

        startTime = std::chrono::steady_clock::now();
        // Shutdown GUI manually since it contains thread. Not strictly necessary but nice to have
        guiManager.reset();
        timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Deleting GUI on exit took {}s", timeSpan.count());

        startTime = std::chrono::steady_clock::now();

        auto view = m_registry.view<entt::any>();
        // Step 3: Destroy entities
        for (auto entity: view) {
            m_registry.destroy(entity);
        }

        timeSpan = std::chrono::duration_cast<std::chrono::duration<float>>(
                std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Deleting entities on exit took {}s", timeSpan.count());


    }


    void Renderer::mouseMoved(float x, float y, bool &handled) {
        float dx = mousePos.x - x;
        float dy = mousePos.y - y;

        mouseButtons.dx = dx;
        mouseButtons.dy = dy;

        bool is3DViewSelected = false;
        for (const auto &dev: guiManager->handles.devices) {
            if (dev.state != VkRender::CRL_STATE_ACTIVE)
                continue;
            is3DViewSelected = dev.selectedPreviewTab == VkRender::CRL_TAB_3D_POINT_CLOUD;
        }

        // UPdate camera if we have one selected
        if (!selectedCameraTag.empty()) {
            auto it = cameras.find(selectedCameraTag);
            if (it != cameras.end()) {
                if (mouseButtons.left && guiManager->handles.info->isViewingAreaHovered &&
                    is3DViewSelected) {
                    // && !mouseButtons.middle) {
                    cameras[selectedCameraTag]->rotate(dx, dy);
                }
                if (mouseButtons.left && guiManager->handles.renderer3D && !guiManager->handles.info->is3DTopBarHovered)
                    cameras[selectedCameraTag]->rotate(dx, dy);

                if (mouseButtons.right) {
                    if (cameras[selectedCameraTag]->m_type == VkRender::Camera::arcball)
                        cameras[selectedCameraTag]->translate(glm::vec3(-dx * 0.005f, -dy * 0.005f, 0.0f));
                    else
                        cameras[selectedCameraTag]->translate(-dx * 0.01f, -dy * 0.01f);
                }
                if (mouseButtons.middle && cameras[selectedCameraTag]->m_type == VkRender::Camera::flycam) {
                    cameras[selectedCameraTag]->translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
                } else if (mouseButtons.middle && cameras[selectedCameraTag]->m_type == VkRender::Camera::arcball) {
                    //camera.orbitPan(static_cast<float>() -dx * 0.01f, static_cast<float>() -dy * 0.01f);
                }
            }
        }
        mousePos = glm::vec2(x, y);

        handled = true;
    }

    void Renderer::mouseScroll(float change) {
        for (const auto &item: guiManager->handles.devices) {
            if (item.state == VkRender::CRL_STATE_ACTIVE &&
                item.selectedPreviewTab == VkRender::CRL_TAB_3D_POINT_CLOUD &&
                guiManager->handles.info->isViewingAreaHovered) {
                cameras[selectedCameraTag]->setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
            }
        }
        if (guiManager->handles.renderer3D) {
            cameras[selectedCameraTag]->setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
        }
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

    std::shared_ptr<Entity> Renderer::createEntitySharedPtr(const std::string &name, UUID uuid) {
        auto entity = std::make_shared<Entity>(m_registry.create(), this);
        entity->addComponent<IDComponent>(UUID());
        entity->addComponent<TransformComponent>();
        auto &tag = entity->addComponent<TagComponent>();
        tag.Tag = name.empty() ? "Entity" : name;

        m_entityMap[uuid] = *entity;  // Store in a map if needed

        return entity;
    }

    VkRender::Entity Renderer::findEntityByName(std::string_view name) {
        auto view = m_registry.view<TagComponent>();
        for (auto entity: view) {
            const TagComponent &tc = view.get<TagComponent>(entity);
            if (tc.Tag == name)
                return Entity{entity, this};
        }
        return {};
    }

    // Destroy when render resources are no longer in use
    void Renderer::markEntityForDestruction(Entity entity) {
        if (!entity.hasComponent<DeleteComponent>()) {
            entity.addComponent<DeleteComponent>();
            Log::Logger::getInstance()->info("Marked Entity for destruction UUID: {} and Tag: {}",
                                             entity.getUUID().operator std::string(), entity.getName());
        }
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

    const VkRender::Camera Renderer::getCamera() {
        if (!selectedCameraTag.empty()) {
            auto it = cameras.find(selectedCameraTag);
            if (it != cameras.end()) {
                return *cameras[selectedCameraTag];
            }
        }
        return Camera(m_Width, m_Height); // Return a non-initialized camera if we dont have any active cameras
    }

    void Renderer::keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {

        cameras[selectedCameraTag]->keys.up = input.keys.up;
        cameras[selectedCameraTag]->keys.down = input.keys.down;
        cameras[selectedCameraTag]->keys.left = input.keys.left;
        cameras[selectedCameraTag]->keys.right = input.keys.right;
    }

    void Renderer::postRenderActions() {
        if (saveDepthPassToFile) {
            auto &secondaryRenderPass = secondaryRenderPasses.front();

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
            barrier.image = secondaryRenderPass.depthStencil.image;
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
                    secondaryRenderPass.depthStencil.image,
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
            float zFar = 25.0f;
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
    void Renderer::onComponentAdded<DeleteComponent>(Entity entity, DeleteComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<GLTFModelComponent>(Entity entity, GLTFModelComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<CustomModelComponent>(Entity entity, CustomModelComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<RenderResource::SkyboxGraphicsPipelineComponent>(Entity entity,
                                                                                     RenderResource::SkyboxGraphicsPipelineComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<RenderResource::DefaultPBRGraphicsPipelineComponent>(Entity entity,
                                                                                         RenderResource::DefaultPBRGraphicsPipelineComponent &component) {
    }


    template<>
    void Renderer::onComponentAdded<VkRender::SecondaryCameraComponent>(Entity entity,
                                                                        VkRender::SecondaryCameraComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<VkRender::OBJModelComponent>(Entity entity,
                                                                 VkRender::OBJModelComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<VkRender::SecondaryRenderPassComponent>(Entity entity,
                                                                            VkRender::SecondaryRenderPassComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<VkRender::ImageViewComponent>(Entity entity,
                                                                  VkRender::ImageViewComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<VkRender::CameraGraphicsPipelineComponent>(Entity entity,
                                                                               VkRender::CameraGraphicsPipelineComponent &component) {
    }

    template<>
    void Renderer::onComponentAdded<VkRender::DefaultGraphicsPipelineComponent2>(Entity entity,
                                                                                 VkRender::DefaultGraphicsPipelineComponent2 &component) {
    }

    DISABLE_WARNING_POP
};