//
// Created by magnus on 7/15/24.
//


#include "Viewer/VkRender/Editor.h"
#include "Viewer/Tools/Utils.h"

#include "Viewer/VkRender/Renderer.h"

#include "Viewer/VkRender/Core/VulkanRenderPass.h"

namespace VkRender {


    Editor::Editor(VulkanRenderPassCreateInfo &createInfo) : m_createInfo(createInfo),
                                                             m_renderUtils(createInfo.context->data()),
                                                             m_context(createInfo.context), sizeLimits(createInfo.appWidth, createInfo.appHeight) {
        borderSize = createInfo.borderSize;

        height = createInfo.height;
        width = createInfo.width;
        x = createInfo.x;
        y = createInfo.y;
        applicationWidth = createInfo.appWidth;
        applicationHeight = createInfo.appHeight;
        uuid = UUID();

        m_renderStates = {createInfo.context->data().swapchainImages, RenderState::Idle};
        renderPasses.resize(m_renderUtils.swapchainImages);
        // Start timing UI render pass setup
        auto startUIRenderPassSetup = std::chrono::high_resolution_clock::now();
        for (auto &pass: renderPasses) {
            pass = std::make_shared<VulkanRenderPass>(createInfo);
        }

        auto endUIRenderPassSetup = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> uiRenderPassSetupTime = endUIRenderPassSetup - startUIRenderPassSetup;
        //std::cout << "UI Render Pass Setup Time: " << uiRenderPassSetupTime.count() << " ms" << std::endl;


        // Start timing GuiManager construction
        auto startGuiManagerConstruction = std::chrono::high_resolution_clock::now();
        m_guiManager = std::make_unique<GuiManager>(m_renderUtils.device,
                                                    renderPasses.begin()->get()->getRenderPass(), // TODO verify if this is ok?
                                                    width,
                                                    height,
                                                    m_renderUtils.msaaSamples,
                                                    m_renderUtils.swapchainImages,
                                                    m_context,
                                                    ImGui::CreateContext(&createInfo.guiResources->fontAtlas),
                                                    createInfo.guiResources.get());

        auto endGuiManagerConstruction = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> guiManagerConstructionTime =
                endGuiManagerConstruction - startGuiManagerConstruction;
        //std::cout << "GuiManager Construction Time: " << guiManagerConstructionTime.count() << " ms" << std::endl;

        std::array<VkImageView, 3> frameBufferAttachments{};
        frameBufferAttachments[0] = createInfo.colorImageView;
        frameBufferAttachments[1] = createInfo.depthImageView;
        VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(applicationWidth,
                                                                                        applicationHeight,
                                                                                        frameBufferAttachments.data(),
                                                                                        frameBufferAttachments.size(),
                                                                                        renderPasses.begin()->get()->getRenderPass() // TODO verify if this is ok?
        );
        // Start timing Framebuffer creation
        std::vector<std::chrono::duration<double, std::milli>> framebufferCreationTimes;
        frameBuffers.resize(m_renderUtils.swapchainImages);
        for (uint32_t i = 0; i < frameBuffers.size(); i++) {
            auto startFramebufferCreation = std::chrono::high_resolution_clock::now();
            frameBufferAttachments[2] = m_context->swapChainBuffers()[i].view;
            VkResult result = vkCreateFramebuffer(m_renderUtils.device->m_LogicalDevice, &frameBufferCreateInfo,
                                                  nullptr, &frameBuffers[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
            auto endFramebufferCreation = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> framebufferCreationTime =
                    endFramebufferCreation - startFramebufferCreation;
            framebufferCreationTimes.push_back(framebufferCreationTime);
            //std::cout << "Framebuffer " << i << " Creation Time: " << framebufferCreationTime.count() << " ms" << std::endl;
        }
        // Optionally, you can average the framebuffer creation times
        double totalFramebufferTime = 0;
        for (const auto &time: framebufferCreationTimes) {
            totalFramebufferTime += time.count();
        }
        //std::cout << "Average Framebuffer Creation Time: " << (totalFramebufferTime / framebufferCreationTimes.size()) << " ms" << std::endl;


    }

    void Editor::render(CommandBuffer &drawCmdBuffers) {
        const uint32_t &currentFrame = *drawCmdBuffers.frameIndex;
        const uint32_t &imageIndex = *drawCmdBuffers.activeImageIndex;

        if (m_renderStates[currentFrame] == RenderState::PendingDeletion)
            return;

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

        renderPassBeginInfo.renderPass = renderPasses[currentFrame]->getRenderPass(); // Increase reference count by 1 here?
        renderPassBeginInfo.renderArea.offset.x = static_cast<int32_t>(x);
        renderPassBeginInfo.renderArea.offset.y = static_cast<int32_t>(y);
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(m_createInfo.clearValue.size());
        renderPassBeginInfo.pClearValues = m_createInfo.clearValue.data();
        renderPassBeginInfo.framebuffer = frameBuffers[imageIndex];
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.renderPassType = RENDER_PASS_UI;

        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);

        m_guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame, width,
                                height, x, y);
        // Draw borders


        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);

        m_renderStates[currentFrame] = RenderState::Busy;
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
                         vmaDestroyImage(m_context->allocator(), objectRenderPass.colorImage.image,
                                         objectRenderPass.colorImage.colorImageAllocation);
                     }
             // Destroy the image and free its memory allocation using VMA
                     if (objectRenderPass.depthStencil.image != VK_NULL_HANDLE) {
                         vmaDestroyImage(m_context->allocator(), objectRenderPass.depthStencil.image,
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
                             vmaDestroyImage(m_context->allocator(), objectRenderPass.colorImage.resolvedImage,
                                             objectRenderPass.colorImage.resolvedImageAllocation);
                         }
                     }

                     vkDestroySampler(m_renderUtils.device->m_LogicalDevice, objectRenderPass.colorImage.sampler, nullptr);
                     vkDestroySampler(m_renderUtils.device->m_LogicalDevice, objectRenderPass.depthStencil.sampler, nullptr);

                     vkDestroyRenderPass(m_renderUtils.device->m_LogicalDevice, objectRenderPass.renderPass, nullptr);

             */
    }

    void Editor::update(bool updateGraph, float frameTime, Input *input) {

        m_guiManager->update(updateGraph, frameTime, width, height, input);

    }

    VulkanRenderPassCreateInfo &Editor::getCreateInfo() {
        return m_createInfo;
    }


    EditorBorderState
    Editor::checkBorderState(const glm::vec2 &mousePos, const MouseButtons buttons, const glm::vec2 &dxdy) const {
        if (!m_createInfo.resizeable)
            return EditorBorderState::None;

        // Check corners first to give them higher priority
        // Top-left corner
        if (mousePos.x >= x && mousePos.x <= x + (borderSize ) && mousePos.y >= y && mousePos.y <= y + (borderSize )) {
            return EditorBorderState::TopLeft;
        }
        // Top-right corner
        if (mousePos.x >= x + width - (borderSize ) && mousePos.x <= x + width && mousePos.y >= y &&
            mousePos.y <= y + (borderSize )) {
            return EditorBorderState::TopRight;
        }
        // Bottom-left corner
        if (mousePos.x >= x && mousePos.x <= x + (borderSize ) && mousePos.y >= y + height - (borderSize ) &&
            mousePos.y <= y + height) {
            return EditorBorderState::BottomLeft;
        }
        // Bottom-right corner
        if (mousePos.x >= x + width - (borderSize ) && mousePos.x <= x + width && mousePos.y >= y + height - (borderSize ) &&
            mousePos.y <= y + height) {
            return EditorBorderState::BottomRight;
        }

        // Check if the mouse position is near the application borders considering the border size
        if (mousePos.x < sizeLimits.MIN_OFFSET_X + borderSize || mousePos.y < sizeLimits.MIN_OFFSET_Y + borderSize ||
            mousePos.x > applicationWidth - sizeLimits.MIN_OFFSET_X - borderSize ||
            mousePos.y > applicationHeight - sizeLimits.MIN_OFFSET_Y - borderSize) {
            return EditorBorderState::None;
        }

        // Check borders
        // Left border
        if (mousePos.x >= x  && mousePos.x <= x + (borderSize ) && mousePos.y >= y && mousePos.y <= y + height) {
            return EditorBorderState::Left;
        }
        // Right border
        if (mousePos.x >= x + width - (borderSize ) && mousePos.x <= x + width  && mousePos.y >= y &&
            mousePos.y <= y + height) {
            return EditorBorderState::Right;
        }
        // Top border
        if (mousePos.x >= x && mousePos.x <= x + width && mousePos.y >= y && mousePos.y <= y + (borderSize )) {
            return EditorBorderState::Top;
        }
        // Bottom border
        if (mousePos.x >= x && mousePos.x <= x + width && mousePos.y >= y + height - (borderSize ) &&
            mousePos.y <= y + height) {
            return EditorBorderState::Bottom;
        }
        // Inside the editor, not on any border
        return EditorBorderState::None;
    }

    EditorBorderState Editor::checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize){
        // Check borders
        if (verticalResize){
            // Top border
            if (mousePos.y >= y && mousePos.y <= y + (borderSize )) {
                return EditorBorderState::Top;
            }
            // Bottom border
            if (mousePos.y >= y + height - (borderSize ) &&
                mousePos.y <= y + height) {
                return EditorBorderState::Bottom;
            }
        } else {
            // Left border
            if (mousePos.x >= x  && mousePos.x <= x + (borderSize )) {
                return EditorBorderState::Left;
            }
            // Right border
            if (mousePos.x >= x + width - (borderSize ) && mousePos.x <= x + width) {
                return EditorBorderState::Right;
            }
        }

        // Inside the editor, not on any border
        return EditorBorderState::None;
    }

    void Editor::validateEditorSize(VulkanRenderPassCreateInfo &createInfo) {
        // Ensure the x offset is within the allowed range
        if (createInfo.x < sizeLimits.MIN_OFFSET_X)
            createInfo.x = sizeLimits.MIN_OFFSET_X;
        if (createInfo.x > sizeLimits.MAX_OFFSET_WIDTH)
            createInfo.x = sizeLimits.MAX_OFFSET_WIDTH;

        // Ensure the y offset is within the allowed range
        if (createInfo.y < sizeLimits.MIN_OFFSET_Y)
            createInfo.y = sizeLimits.MIN_OFFSET_Y;
        if (createInfo.y > sizeLimits.MAX_OFFSET_HEIGHT)
            createInfo.y = sizeLimits.MAX_OFFSET_HEIGHT;

        // Ensure the width is within the allowed range considering the offset
        if (createInfo.width < sizeLimits.MIN_SIZE)
            createInfo.width = sizeLimits.MIN_SIZE;
        if (createInfo.width > applicationWidth - createInfo.x)
            createInfo.width = applicationWidth - createInfo.x - sizeLimits.MIN_OFFSET_X;

        // Ensure the height is within the allowed range considering the offset
        if (createInfo.height < sizeLimits.MIN_SIZE)
            createInfo.height = sizeLimits.MIN_SIZE;
        if (createInfo.height > applicationHeight - createInfo.y)
            createInfo.height = applicationHeight - createInfo.y;
    }

    bool Editor::checkEditorCollision(const Editor& otherEditor) const{
        bool rightSideLeftOfOther = x + width < otherEditor.x; // right side of this is left of other
        bool leftSideRightOfOther = x > otherEditor.x + otherEditor.width; // left side of this is right of other
        bool bottomAboveTopOfOther = y + height < otherEditor.y; // bottom of this is above top of other
        bool topBelowBottomOfOther = y > otherEditor.y + otherEditor.height; // top of this is below bottom of other

        return !(rightSideLeftOfOther || leftSideRightOfOther || bottomAboveTopOfOther || topBelowBottomOfOther);

    }

    const Editor::SizeLimits &Editor::getSizeLimits() const {
        return sizeLimits;
    }
}
