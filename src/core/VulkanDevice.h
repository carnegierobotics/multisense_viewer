//
// Created by magnus on 9/4/21.
//

#ifndef AR_ENGINE_VULKANDEVICE_H
#define AR_ENGINE_VULKANDEVICE_H

#include <cassert>
#include <algorithm>
#include <iostream>
#include <ar_engine/src/tools/Populate.h>
#include "Buffer.h"

struct VulkanDevice {
    /** @brief Physical device representation */
    VkPhysicalDevice physicalDevice{};
    /** @brief Logical device representation (application's view of the device) */
    VkDevice logicalDevice{};
    /** @brief transfer queue for copy operations*/
    VkQueue transferQueue{};
    /** @brief Properties of the physical device including limits that the application can check against */
    VkPhysicalDeviceProperties properties{};
    /** @brief Features of the physical device that an application can use to check if a feature is supported */
    VkPhysicalDeviceFeatures features{};
    /** @brief Features that have been enabled for use on the physical device */
    VkPhysicalDeviceFeatures enabledFeatures{};
    /** @brief Memory types and heaps of the physical device */
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    /** @brief Queue family properties of the physical device */
    std::vector<VkQueueFamilyProperties> queueFamilyProperties{};
    /** @brief List of extensions supported by the device */
    std::vector<std::string> supportedExtensions{};
    /** @brief Default command pool for the graphics queue family index */
    VkCommandPool commandPool = VK_NULL_HANDLE;
    /** @brief Contains queue family indices */
    struct {
        uint32_t graphics{};
        uint32_t compute{};
        uint32_t transfer{};

    } queueFamilyIndices;

    explicit VulkanDevice(VkPhysicalDevice physicalDevice);

    ~VulkanDevice();

    uint32_t
    getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr) const;

    uint32_t getQueueFamilyIndex(VkQueueFlagBits queueFlags) const;

    VkResult
    createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char *> enabledExtensions,
                        void *pNextChain, bool useSwapChain = true,
                        VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);

    VkCommandPool createCommandPool(uint32_t queueFamilyIndex,
                                    VkCommandPoolCreateFlags createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    bool extensionSupported(std::string extension);

    VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size,
                          VkBuffer *buffer, VkDeviceMemory *memory, void *data = nullptr);

    VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, Buffer *buffer,
                          VkDeviceSize size, void *data = nullptr);

    void copyBuffer(Buffer *src, Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion = nullptr);

    void copyVkBuffer(VkBuffer *src, VkBuffer *dst, VkBufferCopy *copyRegion);

    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);

    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin = false);

    void beginCommandBuffer(VkCommandBuffer commandBuffer);

    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);

    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);

};


#endif //AR_ENGINE_VULKANDEVICE_H
