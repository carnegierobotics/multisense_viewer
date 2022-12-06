//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_VULKANSWAPCHAIN_H
#define MULTISENSE_VULKANSWAPCHAIN_H


#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>
#include <cassert>

#include "Viewer/Core/Definitions.h"

typedef struct SwapChainBuffers {
    VkImage image;
    VkImageView view;
} SwapChainBuffer;

class VulkanSwapchain {
private:
    VkInstance instance{};
    VkDevice device{};
    VkPhysicalDevice physicalDevice{};
public:
    VkSurfaceKHR surface{};
    VkFormat colorFormat{};
    VkColorSpaceKHR colorSpace{};
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    uint32_t imageCount=0;
    std::vector<VkImage> images{};
    std::vector<SwapChainBuffer> buffers{};
    uint32_t queueNodeIndex = UINT32_MAX;

    void create(uint32_t* width, uint32_t* height, bool vsync = false);
    VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex);
    VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore = VK_NULL_HANDLE);
    void cleanup();

    VulkanSwapchain(VkRender::SwapChainCreateInfo info, uint32_t *width, uint32_t *height);
};


#endif //MULTISENSE_VULKANSWAPCHAIN_H
