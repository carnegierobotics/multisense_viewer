//
// Created by magnus on 7/15/24.
//


#include "Viewer/VkRender/Editor.h"
#include "Viewer/Tools/Utils.h"

#include "Viewer/VkRender/Renderer.h"

#include "Viewer/VkRender/Core/VulkanRenderPass.h"

namespace VkRender {


    Editor::Editor(VulkanRenderPassCreateInfo &createInfo, UUID _uuid) : m_createInfo(std::move(createInfo)),
                                                                         m_renderUtils(createInfo.context->data()),
                                                                         m_context(createInfo.context),
                                                                         m_sizeLimits(createInfo.appWidth, createInfo.appHeight),
                                                                         m_uuid(_uuid){
        m_ui.borderSize = m_createInfo.borderSize;
        m_ui.height = m_createInfo.height;
        m_ui.width = m_createInfo.width;
        m_ui.x = m_createInfo.x;
        m_ui.y = m_createInfo.y;
        m_ui.index = m_createInfo.editorIndex;

        for (size_t i = 0; i < m_renderUtils.swapchainImages; ++i) {
            m_renderPasses.emplace_back(m_createInfo);
        }


        m_guiManager = std::make_unique<GuiManager>(*m_renderUtils.device,
                                                    m_renderPasses.begin()->getRenderPass(), // TODO verify if this is ok?
                                                    &m_ui,
                                                    m_renderUtils.msaaSamples,
                                                    m_renderUtils.swapchainImages,
                                                    m_context,
                                                    ImGui::CreateContext(&m_createInfo.guiResources->fontAtlas),
                                                    m_createInfo.guiResources.get(),
                                                    m_createInfo.sharedUIContextData);

        Log::Logger::getInstance()->info("Creating new Editor. UUID: {}, size: {}x{}, at pos: ({},{})", m_uuid.operator std::string(), m_ui.width, m_ui.height, m_ui.x, m_ui.y);

    }


    void Editor::resize(VulkanRenderPassCreateInfo &createInfo){
        m_createInfo = std::move(createInfo);
        m_sizeLimits = EditorSizeLimits(m_createInfo.appWidth, m_createInfo.appHeight);

        m_ui.height = m_createInfo.height;
        m_ui.width = m_createInfo.width;
        m_ui.x = m_createInfo.x;
        m_ui.y = m_createInfo.y;

        m_renderPasses.clear();

        for (size_t i = 0; i < m_renderUtils.swapchainImages; ++i) {
            m_renderPasses.emplace_back(m_createInfo);
        }
        m_guiManager->resize(m_ui.width, m_ui.height, m_renderPasses.back().getRenderPass(), m_renderUtils.msaaSamples, m_createInfo.guiResources);

        Log::Logger::getInstance()->info("Resizing Editor. UUID: {}, size: {}x{}, at pos: ({},{})", m_uuid.operator std::string(), m_ui.width, m_ui.height, m_ui.x, m_ui.y);
    }

    void Editor::render(CommandBuffer &drawCmdBuffers) {
        const uint32_t &currentFrame = *drawCmdBuffers.frameIndex;
        const uint32_t &imageIndex = *drawCmdBuffers.activeImageIndex;

        VkViewport viewport{};
        viewport.x = static_cast<float>(m_ui.x);
        viewport.y = static_cast<float>(m_ui.y);
        viewport.width = static_cast<float>(m_ui.width);
        viewport.height = static_cast<float>(m_ui.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {static_cast<int32_t>(m_ui.x), static_cast<int32_t>(m_ui.y)};
        scissor.extent = {static_cast<uint32_t>(m_ui.width), static_cast<uint32_t>(m_ui.height)};
        /// *** Color render pass *** ///
        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = m_renderPasses[currentFrame].getRenderPass(); // Increase reference count by 1 here?
        renderPassBeginInfo.renderArea.offset.x = static_cast<int32_t>(m_ui.x);
        renderPassBeginInfo.renderArea.offset.y = static_cast<int32_t>(m_ui.y);
        renderPassBeginInfo.renderArea.extent.width = m_ui.width;
        renderPassBeginInfo.renderArea.extent.height = m_ui.height;

        renderPassBeginInfo.clearValueCount = 0;
        renderPassBeginInfo.pClearValues = nullptr;

        renderPassBeginInfo.framebuffer = m_createInfo.frameBuffers[imageIndex];
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.renderPassType = RENDER_PASS_UI;
        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);
        m_guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame, m_ui.width,
                                m_ui.height, m_ui.x, m_ui.y);
        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);
    }

    void Editor::update(bool updateGraph, float frameTime, Input *input) {
        m_guiManager->update(updateGraph, frameTime, m_ui, input);
    }

    void Editor::updateBorderState(const glm::vec2 &mousePos) {
        // Check corners first to give them higher priority
        // Top-left corner
        if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + (m_ui.borderSize) && mousePos.y >= m_ui.y && mousePos.y <= m_ui.y + (m_ui.borderSize)) {
            m_ui.lastHoveredBorderType =  EditorBorderState::TopLeft;
            return;
        }
        // Top-right corner
        if (mousePos.x >= m_ui.x + m_ui.width - (m_ui.borderSize) && mousePos.x <= m_ui.x + m_ui.width && mousePos.y >= m_ui.y &&
            mousePos.y <= m_ui.y + (m_ui.borderSize)) {
            m_ui.lastHoveredBorderType =  EditorBorderState::TopRight;
            return;
        }
        // Bottom-left corner
        if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + (m_ui.borderSize) && mousePos.y >= m_ui.y + m_ui.height - (m_ui.borderSize) &&
            mousePos.y <= m_ui.y + m_ui.height) {
            m_ui.lastHoveredBorderType =  EditorBorderState::BottomLeft;
            return;
        }
        // Bottom-right corner
        if (mousePos.x >= m_ui.x + m_ui.width - (m_ui.borderSize) && mousePos.x <= m_ui.x + m_ui.width &&
            mousePos.y >= m_ui.y + m_ui.height - (m_ui.borderSize) &&
            mousePos.y <= m_ui.y + m_ui.height) {
            m_ui.lastHoveredBorderType =  EditorBorderState::BottomRight;
            return;
        }

        // Check borders
        // Left border
        if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + (m_ui.borderSize) && mousePos.y >= m_ui.y && mousePos.y <= m_ui.y + m_ui.height) {
            m_ui.lastHoveredBorderType =  EditorBorderState::Left;
            return;
        }
        // Right border
        if (mousePos.x >= m_ui.x + m_ui.width - (m_ui.borderSize) && mousePos.x <= m_ui.x + m_ui.width && mousePos.y >= m_ui.y &&
            mousePos.y <= m_ui.y + m_ui.height) {
            m_ui.lastHoveredBorderType =  EditorBorderState::Right;
            return;
        }
        // Top border
        if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + m_ui.width && mousePos.y >= m_ui.y && mousePos.y <= m_ui.y + (m_ui.borderSize)) {
            m_ui.lastHoveredBorderType =  EditorBorderState::Top;
            return;
        }
        // Bottom border
        if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + m_ui.width && mousePos.y >= m_ui.y + m_ui.height - (m_ui.borderSize) &&
            mousePos.y <= m_ui.y + m_ui.height) {
            m_ui.lastHoveredBorderType =  EditorBorderState::Bottom;
            return;
        }
        if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + m_ui.width && mousePos.y >= m_ui.y && mousePos.y <= m_ui.y + m_ui.height) {

            m_ui.lastHoveredBorderType = EditorBorderState::Inside;
            return;
        }
        // Outside the editor, not on any border
        m_ui.lastHoveredBorderType =  EditorBorderState::None;
    }

    EditorBorderState Editor::checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize) {
        // Check borders
        if (verticalResize) {
            // Top border
            if (mousePos.y >= m_ui.y && mousePos.y <= m_ui.y + (m_ui.borderSize)) {
                return EditorBorderState::Top;
            }
            // Bottom border
            if (mousePos.y >= m_ui.y + m_ui.height - (m_ui.borderSize) &&
                mousePos.y <= m_ui.y + m_ui.height) {
                return EditorBorderState::Bottom;
            }
        } else {
            // Left border
            if (mousePos.x >= m_ui.x && mousePos.x <= m_ui.x + (m_ui.borderSize)) {
                return EditorBorderState::Left;
            }
            // Right border
            if (mousePos.x >= m_ui.x + m_ui.width - (m_ui.borderSize) && mousePos.x <= m_ui.x + m_ui.width) {
                return EditorBorderState::Right;
            }
        }

        // Inside the editor, not on any border
        return EditorBorderState::None;
    }

    bool Editor::validateEditorSize(VulkanRenderPassCreateInfo &createInfo) {
        // Ensure the x offset is within the allowed range
        if (createInfo.x < m_sizeLimits.MIN_OFFSET_X) {
           return false;
        }
        if (createInfo.x > m_sizeLimits.MAX_OFFSET_WIDTH) {
           return false;
        }
        // Ensure the y offset is within the allowed range
        if (createInfo.y < m_sizeLimits.MIN_OFFSET_Y) {
           return false;
        }
        if (createInfo.y > m_sizeLimits.MAX_OFFSET_HEIGHT) {
           return false;
        }
        // Ensure the width is within the allowed range considering the offset
        if (createInfo.width < m_sizeLimits.MIN_SIZE) {
           return false;
        }
        if (createInfo.width > m_sizeLimits.MAX_WIDTH - createInfo.x) {
            return false;
        }
        // Ensure the height is within the allowed range considering the offset
        if (createInfo.height < m_sizeLimits.MIN_SIZE) {
            return false;
        }
        if (createInfo.height > m_sizeLimits.MAX_HEIGHT) {
            return false;
        }
        return true;
    }

}
