//
// Created by magnus on 7/15/24.
//


#include "Viewer/VkRender/Editor.h"
#include "Viewer/Tools/Utils.h"

#include "Viewer/VkRender/Renderer.h"

#include "Viewer/VkRender/Core/VulkanRenderPass.h"

namespace VkRender {


    Editor::Editor(VulkanRenderPassCreateInfo &createInfo, UUID _uuid) : m_createInfo(createInfo),
                                                                         m_renderUtils(createInfo.context->data()),
                                                                         m_context(createInfo.context),
                                                                         m_sizeLimits(createInfo.appWidth, createInfo.appHeight),
                                                                         m_uuid(_uuid){
        m_ui.borderSize = createInfo.borderSize;
        m_ui.height = createInfo.height;
        m_ui.width = createInfo.width;
        m_ui.x = createInfo.x;
        m_ui.y = createInfo.y;
        m_ui.index = createInfo.editorIndex;
        m_applicationWidth = createInfo.appWidth;
        m_applicationHeight = createInfo.appHeight;
        m_renderStates = {createInfo.context->data().swapchainImages, RenderState::Idle};
        for (size_t i = 0; i < m_renderUtils.swapchainImages; ++i) {
            m_renderPasses.emplace_back(createInfo);
        }

        m_guiManager = std::make_unique<GuiManager>(*m_renderUtils.device,
                                                    m_renderPasses.begin()->getRenderPass(), // TODO verify if this is ok?
                                                    &m_ui,
                                                    m_renderUtils.msaaSamples,
                                                    m_renderUtils.swapchainImages,
                                                    m_context,
                                                    ImGui::CreateContext(&createInfo.guiResources->fontAtlas),
                                                    createInfo.guiResources.get());

        Log::Logger::getInstance()->info("Creating new Editor. UUID: {}, size: {}x{}, at pos: ({},{})", m_uuid.operator std::string(), m_ui.width, m_ui.height, m_ui.x, m_ui.y);

    }

    int32_t Editor::roundToGrid(double value, int gridSize){
        double threshold = 0.1;
        double scaledValue = value / gridSize;
        double roundedValue = std::round(scaledValue);
        if (std::abs(scaledValue - roundedValue) < threshold) {
            return static_cast<int>(roundedValue * gridSize);
        }
        return static_cast<int>(std::round(value / gridSize) * gridSize);
    }



    void Editor::resize(VulkanRenderPassCreateInfo &createInfo){
        m_createInfo = createInfo;
        m_sizeLimits = EditorSizeLimits(createInfo.appWidth, createInfo.appHeight);

        m_ui.height = createInfo.height;
        m_ui.width = createInfo.width;
        m_ui.x = createInfo.x;
        m_ui.y = createInfo.y;
        m_applicationWidth = createInfo.appWidth;
        m_applicationHeight = createInfo.appHeight;

        m_renderPasses.clear();

        for (size_t i = 0; i < m_renderUtils.swapchainImages; ++i) {
            m_renderPasses.emplace_back(createInfo);
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
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(m_createInfo.clearValue.size());
        renderPassBeginInfo.pClearValues = m_createInfo.clearValue.data();
        renderPassBeginInfo.framebuffer = m_createInfo.frameBuffers[imageIndex];
        vkCmdBeginRenderPass(drawCmdBuffers.buffers[currentFrame], &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        drawCmdBuffers.renderPassType = RENDER_PASS_UI;
        vkCmdSetViewport(drawCmdBuffers.buffers[currentFrame], 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers.buffers[currentFrame], 0, 1, &scissor);
        m_guiManager->drawFrame(drawCmdBuffers.buffers[currentFrame], currentFrame, m_ui.width,
                                m_ui.height, m_ui.x, m_ui.y);
        vkCmdEndRenderPass(drawCmdBuffers.buffers[currentFrame]);
        m_renderStates[currentFrame] = RenderState::Busy;
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
        if (createInfo.width > m_applicationWidth - createInfo.x) {
            return false;
        }
        // Ensure the height is within the allowed range considering the offset
        if (createInfo.height < m_sizeLimits.MIN_SIZE) {
            return false;
        }
        if (createInfo.height > m_applicationHeight - createInfo.y) {
            return false;
        }
        return true;
    }

}
