//
// Created by magnus on 9/4/21.
//

#ifndef AR_ENGINE_VULKANSWAPCHAIN_H
#define AR_ENGINE_VULKANSWAPCHAIN_H


#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>

typedef struct _SwapChainBuffers {
    VkImage image;
    VkImageView view;
} SwapChainBuffer;

class VulkanSwapchain {
private:
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkSurfaceKHR surface;
public:
    VkFormat colorFormat;
    VkColorSpaceKHR colorSpace;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    uint32_t imageCount;
    std::vector<VkImage> images;
    std::vector<SwapChainBuffer> buffers;
    uint32_t queueNodeIndex = UINT32_MAX;
    void initSurface(GLFWwindow *window);
    void connect(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);
    void create(uint32_t* width, uint32_t* height, bool vsync = false);
    VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex);
    VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore = VK_NULL_HANDLE);
    void cleanup();

};


#endif //AR_ENGINE_VULKANSWAPCHAIN_H
