//
// Created by magnus on 11/15/23.
//

#ifndef MULTISENSE_VIEWER_COMMANDBUFFER_H
#define MULTISENSE_VIEWER_COMMANDBUFFER_H


#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>

enum RenderPassType {
    RENDER_PASS_COLOR       = 0x00000001,
    RENDER_PASS_DEPTH_ONLY  = 0x00000002,
};

struct CommandBuffer {
    std::vector<VkCommandBuffer> buffers{};
    VkRenderPass boundRenderPass = VK_NULL_HANDLE;
    uint32_t currentFrame = 0;
    RenderPassType renderPassType = RENDER_PASS_COLOR;
};


#endif //MULTISENSE_VIEWER_COMMANDBUFFER_H
