//
// Created by magnus on 11/15/23.
//

#ifndef MULTISENSE_VIEWER_COMMANDBUFFER_H
#define MULTISENSE_VIEWER_COMMANDBUFFER_H


#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>

struct CommandBuffer {
    std::vector<VkCommandBuffer> buffers{};
    VkRenderPass boundRenderPass = VK_NULL_HANDLE;
};


#endif //MULTISENSE_VIEWER_COMMANDBUFFER_H
