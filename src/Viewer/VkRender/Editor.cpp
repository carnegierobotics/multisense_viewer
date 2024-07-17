//
// Created by magnus on 7/15/24.
//


#include "Viewer/VkRender/Editor.h"
#include "Viewer/Tools/Utils.h"

#include "Viewer/VkRender/Renderer.h"

#include "Viewer/VkRender/Core/VulkanRenderPass.h"

namespace VkRender {


    Editor::Editor(const VulkanRenderPassCreateInfo &createInfo) :
    m_renderUtils(createInfo.context.data()), m_context(createInfo.context) {

        width = createInfo.width;
        height = createInfo.height;
        applicationWidth = createInfo.width + createInfo.x;
        applicationHeight = createInfo.height + createInfo.y;

        x = createInfo.x;
        y = createInfo.y;
        editorTypeDescription = createInfo.editorTypeDescription + ":";

        renderPasses.resize(m_renderUtils.swapchainImages);
        // Start timing UI render pass setup
        auto startUIRenderPassSetup = std::chrono::high_resolution_clock::now();
        for (auto& pass : renderPasses){
            pass = std::make_shared<VulkanRenderPass>(createInfo);
        }

        auto endUIRenderPassSetup = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> uiRenderPassSetupTime = endUIRenderPassSetup - startUIRenderPassSetup;
        std::cout << "UI Render Pass Setup Time: " << uiRenderPassSetupTime.count() << " ms" << std::endl;


        // Start timing GuiManager construction
        auto startGuiManagerConstruction = std::chrono::high_resolution_clock::now();
        m_guiManager = std::make_unique<GuiManager>(m_renderUtils.device,
                                                    renderPasses.begin()->get()->renderPass(), // TODO verify if this is ok?
                                                    width,
                                                    height,
                                                    m_renderUtils.msaaSamples,
                                                    m_renderUtils.swapchainImages,
                                                    &m_context, ImGui::CreateContext(&createInfo.guiResources->fontAtlas), createInfo.guiResources.get());

        auto endGuiManagerConstruction = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> guiManagerConstructionTime = endGuiManagerConstruction - startGuiManagerConstruction;
        std::cout << "GuiManager Construction Time: " << guiManagerConstructionTime.count() << " ms" << std::endl;

        std::array<VkImageView, 3> frameBufferAttachments{};
        frameBufferAttachments[0] = createInfo.colorImageView;
        frameBufferAttachments[1] = createInfo.depthImageView;
        VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(applicationWidth,
                                                                                        applicationHeight,
                                                                                        frameBufferAttachments.data(),
                                                                                        frameBufferAttachments.size(),
                                                                                        renderPasses.begin()->get()->renderPass() // TODO verify if this is ok?
                                                                                        );
        // Start timing Framebuffer creation
        std::vector<std::chrono::duration<double, std::milli>> framebufferCreationTimes;
        frameBuffers.resize(m_renderUtils.swapchainImages);
        for (uint32_t i = 0; i < frameBuffers.size(); i++) {
            auto startFramebufferCreation = std::chrono::high_resolution_clock::now();
            frameBufferAttachments[2] = m_context.swapChainBuffers()[i].view;
            VkResult result = vkCreateFramebuffer(m_renderUtils.device->m_LogicalDevice, &frameBufferCreateInfo,
                                                  nullptr, &frameBuffers[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
            auto endFramebufferCreation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> framebufferCreationTime = endFramebufferCreation - startFramebufferCreation;
            framebufferCreationTimes.push_back(framebufferCreationTime);
            std::cout << "Framebuffer " << i << " Creation Time: " << framebufferCreationTime.count() << " ms" << std::endl;
        }
        // Optionally, you can average the framebuffer creation times
        double totalFramebufferTime = 0;
        for (const auto& time : framebufferCreationTimes) {
            totalFramebufferTime += time.count();
        }
        std::cout << "Average Framebuffer Creation Time: " << (totalFramebufferTime / framebufferCreationTimes.size()) << " ms" << std::endl;


    }

    void Editor::render(CommandBuffer& drawCmdBuffers) {

        const uint32_t& currentFrame =  *drawCmdBuffers.frameIndex;
        const uint32_t& imageIndex =    *drawCmdBuffers.activeImageIndex;
        // main editor window
        VkViewport viewport{};
        viewport.x = static_cast<float>(x);
        viewport.y = static_cast<float>(y);
        viewport.width = static_cast<float>(width);
        viewport.height = static_cast<float>(height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
        scissor.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        // Begin render pass
        /// *** Color render pass *** ///
        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        std::array<VkClearValue, 3> clearValues{};
        clearValues[0] = {{{0.1f, 0.1f, 0.3f, 1.0f}}};
        renderPassBeginInfo.renderPass = renderPasses[currentFrame]->renderPass();
        renderPassBeginInfo.renderArea.offset.x = static_cast<int32_t>(x);
        renderPassBeginInfo.renderArea.offset.y = static_cast<int32_t>(y);
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        renderPassBeginInfo.framebuffer = frameBuffers[imageIndex];
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.boundRenderPass = renderPassBeginInfo.renderPass;
        drawCmdBuffers.renderPassType = RENDER_PASS_UI;
        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);

        m_guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame, width,
                                height, x, y);

        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);

    }

    Editor::~Editor() {

        for (auto &fb: frameBuffers) {
            vkDestroyFramebuffer(m_renderUtils.device->m_LogicalDevice, fb, nullptr);
        }


        /*
        // Destroy the image view
        if (objectRenderPass.colorImage.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_renderUtils.device->m_LogicalDevice, objectRenderPass.colorImage.view, nullptr);
        }

// Destroy the image and free its memory allocation using VMA
        if (objectRenderPass.colorImage.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_context.allocator(), objectRenderPass.colorImage.image,
                            objectRenderPass.colorImage.colorImageAllocation);
        }
// Destroy the image and free its memory allocation using VMA
        if (objectRenderPass.depthStencil.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_context.allocator(), objectRenderPass.depthStencil.image,
                            objectRenderPass.depthStencil.allocation);
        }

        // Destroy the resolved image view
        if (objectRenderPass.depthStencil.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_renderUtils.device->m_LogicalDevice, objectRenderPass.depthStencil.view,
                               nullptr);
        }

        if (objectRenderPass.multisampled) {
            // Destroy the resolved image view
            if (objectRenderPass.colorImage.resolvedView != VK_NULL_HANDLE) {
                vkDestroyImageView(m_renderUtils.device->m_LogicalDevice, objectRenderPass.colorImage.resolvedView,
                                   nullptr);
            }


            // Destroy the resolved image and free its memory allocation using VMA
            if (objectRenderPass.colorImage.resolvedImage != VK_NULL_HANDLE) {
                vmaDestroyImage(m_context.allocator(), objectRenderPass.colorImage.resolvedImage,
                                objectRenderPass.colorImage.resolvedImageAllocation);
            }
        }

        vkDestroySampler(m_renderUtils.device->m_LogicalDevice, objectRenderPass.colorImage.sampler, nullptr);
        vkDestroySampler(m_renderUtils.device->m_LogicalDevice, objectRenderPass.depthStencil.sampler, nullptr);

        vkDestroyRenderPass(m_renderUtils.device->m_LogicalDevice, objectRenderPass.renderPass, nullptr);

*/
    }

    void Editor::update(bool updateGraph, float frameTime, Input* input) {

        m_guiManager->update(updateGraph, frameTime, width, height, input);

    }
}
