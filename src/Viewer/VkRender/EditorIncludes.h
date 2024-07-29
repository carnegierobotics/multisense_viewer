//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORINCLUDES_H
#define MULTISENSE_VIEWER_EDITORINCLUDES_H

#include <imgui.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "Viewer/VkRender/pch.h"

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
        BottomRight = 8,  // Cursor is on the bottom-right corner
        Inside = 9  // Cursor is on the bottom-right corner
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
        case VkRender::Inside:
            os << "Inside";
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

    struct EditorSizeLimits {
        const int MENU_BAR_HEIGHT = 25;

        const int MIN_SIZE = 20;
        const int MIN_OFFSET_X = 0;
        const int MIN_OFFSET_Y = MENU_BAR_HEIGHT;
        int MAX_OFFSET_WIDTH;
        int MAX_OFFSET_HEIGHT;
        int MAX_WIDTH;
        int MAX_HEIGHT;

        EditorSizeLimits(uint32_t appWidth, uint32_t appHeight) {
            MAX_OFFSET_WIDTH = appWidth - MIN_SIZE;
            MAX_OFFSET_HEIGHT = appHeight - MIN_SIZE;

            MAX_WIDTH = appWidth;
            MAX_HEIGHT = appHeight - MENU_BAR_HEIGHT;
        }
    };

    struct EditorUI {

        // Todo make private
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
        bool dragHorizontal = false;  // Holding after click event
        bool dragVertical = false;  // Holding after click event
        bool dragActive = false;  // Holding after click event
        bool createNewEditorByCopy = false;
        bool splitting = false;
        EditorBorderState lastClickedBorderType = EditorBorderState::None;
        EditorBorderState lastHoveredBorderType = EditorBorderState::None;
        glm::ivec2 lastPressedPos{};
        glm::ivec2 hoverDelta{};
        glm::ivec2 dragDelta{};
        glm::ivec2 resizeIntervalHoriz{};
        glm::ivec2 resizeIntervalVertical{};

        ImVec4 backgroundColor{};
        ImVec4 backgroundColorHovered{};
        ImVec4 backgroundColorActive{};

        bool active = false;
        bool hovered = false;

        // assign some random background colors
        EditorUI(){
            std::vector<ImVec4> c{{1.0f, 0.0f, 0.0f, 0.6f},
                                  {0.0f, 1.0f, 0.0f, 0.6f}, // 1: Green
                                  {0.0f, 0.0f, 1.0f, 0.6f}, // 2: Blue
                                  {1.0f, 1.0f, 0.0f, 0.6f}, // 3: Yellow
                                  {1.0f, 0.0f, 1.0f, 0.6f}, // 4: Magenta
                                  {0.0f, 1.0f, 1.0f, 0.6f}, // 5: Cyan
                                  {0.5f, 0.5f, 0.5f, 0.6f}}; // 6: Gray
            std::random_device rd;  // Seed for the random number engine
            std::mt19937 gen(rd()); // Mersenne Twister random number engine
            std::uniform_int_distribution<> dis(0, c.size() - 1); // Uniform distribution
            size_t index = dis(gen); // Generate random index

            backgroundColor = {c[index].x, c[index].y, c[index].z, c[index].w};
            backgroundColorHovered = backgroundColor;
            backgroundColorHovered.w = 0.75f;
            backgroundColorActive = backgroundColor;
            backgroundColorActive.w = 1.0f;
        }
    };

}
#endif //MULTISENSE_VIEWER_EDITORINCLUDES_H
