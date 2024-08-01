/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Core/VulkanDevice.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2021-09-4, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VULKANDEVICE_H
#define MULTISENSE_VULKANDEVICE_H

#include "Viewer/VkRender/pch.h"

#include "Viewer/Tools/Populate.h"
#include "Viewer/VkRender/Core/Buffer.h"

struct VulkanDevice {

/** @brief Physical m_Device representation */
    VkPhysicalDevice m_PhysicalDevice{};
    /** @brief Logical m_Device representation (application's m_View of the m_Device) */
    VkDevice m_LogicalDevice{};
    /** @brief transfer queue for copy operations*/
    VkQueue m_TransferQueue{};
    /** @brief synchronozation if vkQueueSubmit is run from thread */
    std::mutex *m_QueueSubmitMutex;
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


    VulkanDevice(VkPhysicalDevice physicalDevice, std::mutex *mut);
    explicit VulkanDevice(VulkanDevice* copy);
    bool isCopy = false;
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
                          VkBuffer *buffer, VkDeviceMemory *memory, const void *data = nullptr) const;

    VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, Buffer *buffer,
                          VkDeviceSize size, void *data = nullptr);

    //void copyBuffer(Buffer *src, Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion = nullptr);

    void copyVkBuffer(VkBuffer *src, VkBuffer *dst, VkBufferCopy *copyRegion);

    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);

    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin = false);

    void beginCommandBuffer(VkCommandBuffer commandBuffer);

    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);

    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);

    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free, VkFence &fence);
};


#endif //MULTISENSE_VULKANDEVICE_H
