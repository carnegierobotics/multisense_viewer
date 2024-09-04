//
// Created by magnus on 11/15/23.
//

#ifndef MULTISENSE_VIEWER_COMMANDBUFFER_H
#define MULTISENSE_VIEWER_COMMANDBUFFER_H


#include "Viewer/Application/pch.h"

#include <vulkan/vulkan_core.h>

enum RenderPassType {
    RENDER_PASS_COLOR       = 0x00000001,
    RENDER_PASS_DEPTH_ONLY  = 0x00000002,
    RENDER_PASS_UI          = 0x00000004,
    RENDER_PASS_SECOND          = 0x00000008,
};

struct CommandBuffer {
    std::vector<VkCommandBuffer> buffers{};

    std::vector<VkRenderPass> boundRenderPasses;

    uint32_t* frameIndex = nullptr;
    uint32_t* activeImageIndex = nullptr;
    RenderPassType renderPassType = RENDER_PASS_COLOR;
};


#endif //MULTISENSE_VIEWER_COMMANDBUFFER_H
