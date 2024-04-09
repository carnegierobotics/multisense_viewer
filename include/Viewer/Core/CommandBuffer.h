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
    std::vector<bool> hasWork{};

    // Owner --> Description?
    std::string description = "RenderCommandBuffer";

    std::string boundRenderPass = "main";
    std::vector<bool> busy{};

};


#endif //MULTISENSE_VIEWER_COMMANDBUFFER_H
