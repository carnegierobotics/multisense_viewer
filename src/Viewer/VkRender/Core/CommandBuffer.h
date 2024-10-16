//
// Created by magnus on 11/15/23.
//

#ifndef MULTISENSE_VIEWER_COMMANDBUFFER_H
#define MULTISENSE_VIEWER_COMMANDBUFFER_H


#include "Viewer/Application/pch.h"

#include <vulkan/vulkan_core.h>

enum RenderPassType {
    RENDER_PASS_COLOR = 0x00000001,
    RENDER_PASS_DEPTH_ONLY = 0x00000002,
    RENDER_PASS_UI = 0x00000004,
    RENDER_PASS_SECOND = 0x00000008,
};

struct CommandBuffer {
    uint32_t frameIndex = 0; // Current frame index (e.g., for triple buffering)
    uint32_t activeImageIndex = 0; // Current active image index (for swapchain images)
    // Constructor
    CommandBuffer() = default;

    explicit CommandBuffer(uint32_t frameCount) {
        buffers.resize(frameCount);
    }

    // Function to get the active buffer
    VkCommandBuffer getActiveBuffer() const {
        if (frameIndex < buffers.size()) {
            return buffers[frameIndex];
        }
        throw std::runtime_error("Invalid frame index for command buffer!");
    } // Function to get the active buffer

    std::vector<VkCommandBuffer>& getBuffers() {
        return buffers;
    }
private:
    std::vector<VkCommandBuffer> buffers;
};


#endif //MULTISENSE_VIEWER_COMMANDBUFFER_H
