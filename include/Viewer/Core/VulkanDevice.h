//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_VULKANDEVICE_H
#define MULTISENSE_VULKANDEVICE_H

#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

#include <Viewer/Tools/Populate.h>
#include "Viewer/Core/Buffer.h"

struct VulkanDevice {
    /** @brief Physical m_Device representation */
    VkPhysicalDevice m_PhysicalDevice{};
    /** @brief Logical m_Device representation (application's m_View of the m_Device) */
    VkDevice m_LogicalDevice{};
    /** @brief transfer queue for copy operations*/
    VkQueue m_TransferQueue{};
    /** @brief Properties of the physical m_Device including limits that the application can check against */
    VkPhysicalDeviceProperties m_Properties{};
    /** @brief Features of the physical m_Device that an application can use to check if a feature is supported */
    VkPhysicalDeviceFeatures m_Features{};
    /** @brief Features that have been enabled for use on the physical m_Device */
    VkPhysicalDeviceFeatures m_EnabledFeatures{};
    /** @brief Memory types and heaps of the physical m_Device */
    VkPhysicalDeviceMemoryProperties m_MemoryProperties{};
    /** @brief Queue family m_Properties of the physical m_Device */
    std::vector<VkQueueFamilyProperties> m_QueueFamilyProperties{};
    /** @brief List of extensions supported by the m_Device */
    std::vector<std::string> m_SupportedExtensions{};
    /** @brief Default command pool for the graphics queue family index */
    VkCommandPool m_CommandPool = VK_NULL_HANDLE;
    /** @brief Contains queue family indices */
    struct {
        uint32_t graphics{};
        uint32_t compute{};
        uint32_t transfer{};

    } m_QueueFamilyIndices;

    explicit VulkanDevice(VkPhysicalDevice physicalDevice);

    ~VulkanDevice();

    uint32_t
    getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags propertyFlags, VkBool32 *memTypeFound = nullptr) const;

    uint32_t getQueueFamilyIndex(VkQueueFlagBits queueFlags) const;

    VkResult
    createLogicalDevice(VkPhysicalDeviceFeatures enabled, std::vector<const char *> enabledExtensions,
                        void *pNextChain, bool useSwapChain = true,
                        VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);

    VkCommandPool createCommandPool(uint32_t queueFamilyIndex,
                                    VkCommandPoolCreateFlags createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    bool extensionSupported(std::string extension) const;

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


#endif //MULTISENSE_VULKANDEVICE_H
