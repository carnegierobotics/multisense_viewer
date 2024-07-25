//
// Created by magnus on 7/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR_H
#define MULTISENSE_VIEWER_EDITOR_H

#include <vk_mem_alloc.h>
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/VulkanRenderPass.h"
#include "Viewer/VkRender/ImGui/GuiManager.h"

namespace VkRender {
    class Renderer;


    class Editor {
    public:

        typedef enum EditorBorderState {
            None,        // Cursor is not on any border
            Left,        // Cursor is on the left border
            Right,       // Cursor is on the right border
            Top,         // Cursor is on the top border
            Bottom,      // Cursor is on the bottom border
            TopLeft,     // Cursor is on the top-left corner
            TopRight,    // Cursor is on the top-right corner
            BottomLeft,  // Cursor is on the bottom-left corner
            BottomRight  // Cursor is on the bottom-right corner
        } EditorBorderState;

        explicit Editor(VulkanRenderPassCreateInfo &createInfo);

        // Implement move constructor and move assignment operator
        Editor(Editor &&other) noexcept: m_context(other.m_context), m_renderUtils(other.m_renderUtils),
                                         m_renderStates(other.m_renderStates), m_createInfo(other.m_createInfo) {
            std::cout << "Move constructor\n";
        }

        Editor &operator=(Editor &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                swap(*this, other);
                std::cout << "Move assignment operator\n";
            }
            return *this;
        }

        // No copying allowed
        Editor(const Editor &) = delete;

        Editor &operator=(const Editor &) = delete;


        // Implement a swap function
        friend void swap(Editor &first, Editor &second) noexcept {
            using std::swap;
            swap(first.m_renderStates, second.m_renderStates);
            // Swap other members
            // TODO implement
        }

        ~Editor();

        bool isSafeToDelete(size_t index) const {
            return m_renderStates[index] == RenderState::Idle;
        }

        void setRenderState(size_t index, RenderState state) {
            m_renderStates[index] = state;
        }

        RenderState getRenderState(size_t index) const {
            return m_renderStates[index];
        }

        VulkanRenderPassCreateInfo &getCreateInfo();

        void render(CommandBuffer &drawCmdBuffers);

        void update(bool updateGraph, float frametime, Input *input);

        EditorBorderState checkBorderState(const glm::vec2 &mousePos, const MouseButtons buttons, const glm::vec2 &dxdy) const;


        // Todo make private
        std::unique_ptr<GuiManager> m_guiManager;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t borderSize = 0;

        uint32_t width = 0;
        uint32_t height = 0;
        bool resizeActive = false;
        bool prevResize = false;
        bool borderClicked = false;
        bool resizeHovered = false;
        bool cornerHovered = false;
        bool cornerClicked = false;
        bool createNewEditorByCopy=false;

        glm::vec2 cornerPressedPos;
    private:

        std::vector<std::shared_ptr<VulkanRenderPass>> renderPasses;
        RenderUtils &m_renderUtils;
        Renderer *m_context;

        uint32_t applicationWidth = 0;
        uint32_t applicationHeight = 0;

        std::vector<VkFramebuffer> frameBuffers;
        std::vector<RenderState> m_renderStates;  // States for each swapchain image
        VulkanRenderPassCreateInfo m_createInfo;
    };
}

#endif //MULTISE{}NSE_VIEWER_EDITOR_H
