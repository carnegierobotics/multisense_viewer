//
// Created by magnus on 7/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR_H
#define MULTISENSE_VIEWER_EDITOR_H

#include <vk_mem_alloc.h>

#include <iostream>
#include <string>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/VulkanRenderPass.h"
#include "Viewer/VkRender/ImGui/GuiManager.h"
#include "Viewer/VkRender/Core/UUID.h"

namespace VkRender {


    typedef enum EditorBorderState {
        None = 0,        // Cursor is not on any border
        Left = 1,        // Cursor is on the left border
        Right = 2,       // Cursor is on the right border
        Top = 3,         // Cursor is on the top border
        Bottom = 4,      // Cursor is on the bottom border
        TopLeft = 5,     // Cursor is on the top-left corner
        TopRight = 6,    // Cursor is on the top-right corner
        BottomLeft = 7,  // Cursor is on the bottom-left corner
        BottomRight = 8  // Cursor is on the bottom-right corner
    } EditorBorderState;
}

// Overload the << operator for EditorBorderState
static std::ostream &operator<<(std::ostream &os, const VkRender::EditorBorderState &state) {
    switch (state) {
        case VkRender::None:
            os << "None";
            break;
        case VkRender::Left:
            os << "Left";
            break;
        case VkRender::Right:
            os << "Right";
            break;
        case VkRender::Top:
            os << "Top";
            break;
        case VkRender::Bottom:
            os << "Bottom";
            break;
        case VkRender::TopLeft:
            os << "TopLeft";
            break;
        case VkRender::TopRight:
            os << "TopRight";
            break;
        case VkRender::BottomLeft:
            os << "BottomLeft";
            break;
        case VkRender::BottomRight:
            os << "BottomRight";
            break;
        default:
            os << "Unknown";
            break;
    }
    return os;
}

// Define a custom formatter for EditorBorderState for fmt within the fmt namespace
namespace fmt {
    template<>
    struct formatter<VkRender::EditorBorderState> : ostream_formatter {
    };
}

namespace VkRender {
    class Renderer;


    class Editor {
    public:

        struct SizeLimits {
            const int MENU_BAR_HEIGHT = 25;

            const int MIN_SIZE = 20;
            const int MIN_OFFSET_X = 0;
            const int MIN_OFFSET_Y = MENU_BAR_HEIGHT;
            int MAX_OFFSET_WIDTH;
            int MAX_OFFSET_HEIGHT;

            SizeLimits(uint32_t appWidth, uint32_t appHeight) {
                MAX_OFFSET_WIDTH = appWidth - MIN_SIZE;
                MAX_OFFSET_HEIGHT = appHeight - MIN_SIZE;
            }
        };


        explicit Editor(VulkanRenderPassCreateInfo &createInfo);

        // Implement move constructor
        Editor(Editor &&other) noexcept: m_context(other.m_context), m_renderUtils(other.m_renderUtils),
                                         m_renderStates(other.m_renderStates), m_createInfo(other.m_createInfo),
                                         sizeLimits(other.m_createInfo.appWidth, other.m_createInfo.appHeight) {
            swap(*this, other);
        }

        // and move assignment operator
        Editor &operator=(Editor &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                swap(*this, other);
            }
            return *this;
        }

        // No copying allowed
        Editor(const Editor &) = delete;
        Editor &operator=(const Editor &) = delete;

        // Implement a swap function
        friend void swap(Editor &first, Editor &second) noexcept {

            std::swap(first.m_guiManager, second.m_guiManager);
            std::swap(first.x, second.x);
            std::swap(first.y, second.y);
            std::swap(first.borderSize, second.borderSize);
            std::swap(first.width, second.width);
            std::swap(first.height, second.height);
            std::swap(first.resizeActive, second.resizeActive);
            std::swap(first.prevResize, second.prevResize);
            std::swap(first.borderClicked, second.borderClicked);
            std::swap(first.resizeHovered, second.resizeHovered);
            std::swap(first.cornerBottomLeftHovered, second.cornerBottomLeftHovered);
            std::swap(first.cornerBottomLeftClicked, second.cornerBottomLeftClicked);
            std::swap(first.createNewEditorByCopy, second.createNewEditorByCopy);
            std::swap(first.lastClickedBorderType, second.lastClickedBorderType);
            std::swap(first.cornerPressedPos, second.cornerPressedPos);

            std::swap(first.renderPasses, second.renderPasses);
            std::swap(first.m_renderUtils, second.m_renderUtils);
            std::swap(first.m_context, second.m_context);
            std::swap(first.applicationWidth, second.applicationWidth);
            std::swap(first.applicationHeight, second.applicationHeight);
            std::swap(first.frameBuffers, second.frameBuffers);
            std::swap(first.m_renderStates, second.m_renderStates);
            std::swap(first.m_createInfo, second.m_createInfo);
        }

        // Comparison operator
        bool operator==(const Editor& other) const {
            return other.getUUID() == getUUID();
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

        EditorBorderState
        checkBorderState(const glm::vec2 &mousePos, const MouseButtons buttons, const glm::vec2 &dxdy) const;
        EditorBorderState checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize);

        void validateEditorSize(VulkanRenderPassCreateInfo &createInfo);
        bool checkEditorCollision(const Editor &otherEditor) const;
        UUID getUUID() const {return uuid;}

        const SizeLimits &getSizeLimits() const;

        // Todo make private
        std::unique_ptr<GuiManager> m_guiManager;
        int32_t x = 0;
        int32_t y = 0;
        uint32_t borderSize = 0;

        int32_t width = 0;
        int32_t height = 0;
        bool resizeActive = false;
        bool prevResize = false;
        bool borderClicked = false;
        bool resizeHovered = false;
        bool cornerBottomLeftHovered = false; // Hover state
        bool cornerBottomLeftClicked = false; // Single push down event
        bool cornerBottomLeftDragEvent = false;  // Holding after click event
        bool createNewEditorByCopy = false;
        EditorBorderState lastClickedBorderType;

        glm::vec2 cornerPressedPos{};
    private:
        UUID uuid;
        std::vector<std::shared_ptr<VulkanRenderPass>> renderPasses;
        RenderUtils &m_renderUtils;
        Renderer *m_context;

        uint32_t applicationWidth = 0;
        uint32_t applicationHeight = 0;
        SizeLimits sizeLimits;

        std::vector<VkFramebuffer> frameBuffers;
        std::vector<RenderState> m_renderStates;  // States for each swapchain image
        VulkanRenderPassCreateInfo m_createInfo;

        void handeViewportResize();

    };
}

#endif //MULTISE{}NSE_VIEWER_EDITOR_H
