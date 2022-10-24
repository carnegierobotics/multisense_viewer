//
// Created by magnus on 9/5/21.
//

#include "VulkanDevice.h"
#include <MultiSense/Src/Tools/Macros.h>

#include <utility>


VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice) {
    assert(m_PhysicalDevice);
    this->m_PhysicalDevice = physicalDevice;
    // Store property m_Features and such for the m_Device. Can be used for later
    // Device m_Properties also contain limits and sparse m_Properties
    vkGetPhysicalDeviceProperties(physicalDevice, &m_Properties);
    // Features should be checked by the examples before using them
    vkGetPhysicalDeviceFeatures(physicalDevice, &m_Features);
    // Memory m_Properties are used regularly for creating all kinds of buffers
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &m_MemoryProperties);
    // Queue family m_Properties, used for setting up requested queues upon m_Device creation
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    assert(queueFamilyCount > 0);
    m_QueueFamilyProperties.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, m_QueueFamilyProperties.data());

    // Get list of supported extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    if (extCount > 0) {
        std::vector<VkExtensionProperties> extensions(extCount);
        if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, &extensions.front()) ==
            VK_SUCCESS) {
            for (const auto& ext: extensions) {
                m_SupportedExtensions.push_back(ext.extensionName);
            }
        }
    }


}

VulkanDevice::~VulkanDevice() {
    if (m_CommandPool) {
        vkDestroyCommandPool(m_LogicalDevice, m_CommandPool, nullptr);
    }
    if (m_LogicalDevice) {
        vkDestroyDevice(m_LogicalDevice, nullptr);
    }
}

/**
	* Get the index of a memory type that has all the requested property bits set
	*
	* @param typeBits Bit mask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
	* @param propertyFlags Bit mask of propertyFlags for the memory type to request
	* @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
	*
	* @return Index of the requested memory type
	*
	* @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested propertyFlags
	*/
uint32_t
VulkanDevice::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags propertyFlags, VkBool32 *memTypeFound) const {
    for (uint32_t i = 0; i < m_MemoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            if ((m_MemoryProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags) {
                if (memTypeFound) {
                    *memTypeFound = true;
                }
                return i;
            }
        }
        typeBits >>= 1;
    }

    if (memTypeFound) {
        *memTypeFound = false;
        return 0;
    } else {
        throw std::runtime_error("Could not find a matching memory type");
    }
}

/**
	* Get the index of a queue family that supports the requested queue flags
	*
	* @param queueFlags Queue flags to find a queue family index for
	*
	* @return Index of the queue family index that matches the flags
	*
	* @throw Throws an exception if no queue family index could be found that supports the requested flags
	*/
uint32_t VulkanDevice::getQueueFamilyIndex(VkQueueFlagBits queueFlags) const {
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if (queueFlags & VK_QUEUE_COMPUTE_BIT) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_QueueFamilyProperties.size()); i++) {
            if ((m_QueueFamilyProperties[i].queueFlags & queueFlags) &&
                ((m_QueueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)) {
                return i;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if (queueFlags & VK_QUEUE_TRANSFER_BIT) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_QueueFamilyProperties.size()); i++) {
            if ((m_QueueFamilyProperties[i].queueFlags & queueFlags) &&
                ((m_QueueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) &&
                ((m_QueueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)) {
                return i;
            }
        }
    }

    // For other queue types or if no separate compute queue is present, return the first one to support the requested flags
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_QueueFamilyProperties.size()); i++) {
        if (m_QueueFamilyProperties[i].queueFlags & queueFlags) {
            return i;
        }
    }

    throw std::runtime_error("Could not find a matching queue family index");
}

/**
	* Create the logical m_Device based on the assigned physical m_Device, also gets default queue family indices
	*
	* @param enabled Can be used to flashing certain m_Features upon m_Device creation
	* @param pNextChain Optional chain of pointer to extension structures
	* @param useSwapChain Set to false for headless rendering to omit the swapchain m_Device extensions
	* @param requestedQueueTypes Bit flags specifying the queue types to be requested from the m_Device
	*
	* @return VkResult of the m_Device creation call
	*/
VkResult
VulkanDevice::createLogicalDevice(VkPhysicalDeviceFeatures enabled, std::vector<const char *> enabledExtensions,
                                  void *pNextChain, bool useSwapChain, VkQueueFlags requestedQueueTypes) {
    // Desired queues need to be requested upon logical m_Device creation
    // Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
    // requests different queue types

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};

    // Get queue family indices for the requested queue family types
    // Note that the indices may overlap depending on the implementation

    const float defaultQueuePriority(0.0f);

    // Graphics queue
    if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT) {
        m_QueueFamilyIndices.graphics = getQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT);
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = m_QueueFamilyIndices.graphics;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &defaultQueuePriority;
        queueCreateInfos.push_back(queueInfo);
    } else {
        m_QueueFamilyIndices.graphics = 0;
    }

    // Dedicated compute queue
    if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT) {
        m_QueueFamilyIndices.compute = getQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT);
        if (m_QueueFamilyIndices.compute != m_QueueFamilyIndices.graphics) {
            // If compute family index differs, we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = m_QueueFamilyIndices.compute;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &defaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    } else {
        // Else we use the same queue
        m_QueueFamilyIndices.compute = m_QueueFamilyIndices.graphics;
    }

    // Dedicated transfer queue
    if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT) {
        m_QueueFamilyIndices.transfer = getQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT);
        if ((m_QueueFamilyIndices.transfer != m_QueueFamilyIndices.graphics) &&
            (m_QueueFamilyIndices.transfer != m_QueueFamilyIndices.compute)) {
            // If compute family index differs, we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = m_QueueFamilyIndices.transfer;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &defaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    } else {
        // Else we use the same queue
        m_QueueFamilyIndices.transfer = m_QueueFamilyIndices.graphics;
    }

    // Create the logical m_Device representation
    std::vector<const char *> deviceExtensions(std::move(enabledExtensions));
    if (useSwapChain) {
        // If the m_Device will be used for presenting to a display via a swapchain we need to request the swapchain extension
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = &enabled;

    // If a pNext(Chain) has been passed, we need to add it to the m_Device creation info
    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    if (pNextChain) {
        physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        physicalDeviceFeatures2.features = enabled;
        physicalDeviceFeatures2.pNext = pNextChain;
        deviceCreateInfo.pEnabledFeatures = nullptr;
        deviceCreateInfo.pNext = &physicalDeviceFeatures2;
    }



    // Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
    if (extensionSupported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        deviceExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
    }


    if (deviceExtensions.size() > 0) {
        for (const char *enabledExtension: deviceExtensions) {
            if (!extensionSupported(enabledExtension)) {
                std::cerr << "Enabled m_Device extension \"" << enabledExtension << "\" is not present at m_Device level\n";
            }
        }

        deviceCreateInfo.enabledExtensionCount = (uint32_t) deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }

    this->m_EnabledFeatures = enabled;

    VkResult result = vkCreateDevice(m_PhysicalDevice, &deviceCreateInfo, nullptr, &m_LogicalDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Create a default command pool for graphics command buffers
    m_CommandPool = createCommandPool(m_QueueFamilyIndices.graphics);

    // Initialize a transfer queue
    vkGetDeviceQueue(m_LogicalDevice, m_QueueFamilyIndices.transfer, 0, &m_TransferQueue);

    return result;


}

/**
	* Check if an extension is supported by the (physical m_Device)
	*
	* @param extension Name of the extension to check
	*
	* @return True if the extension is supported (present in the list read at m_Device creation time)
	*/
bool VulkanDevice::extensionSupported(std::string extension) {
    return (std::find(m_SupportedExtensions.begin(), m_SupportedExtensions.end(), extension) != m_SupportedExtensions.end());
}


/**
* Create a command pool for allocation command buffers from
*
* @param queueFamilyIndex Family index of the queue to create the command pool for
* @param createFlags (Optional) Command pool creation flags (Defaults to VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
*
* @note Command buffers allocated from the created pool can only be submitted to a queue with the same family index
*
* @return A handle to the created command buffer
*/
VkCommandPool VulkanDevice::createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags) {
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
    cmdPoolInfo.flags = createFlags;
    VkCommandPool cmdPool;
    VkResult result = vkCreateCommandPool(m_LogicalDevice, &cmdPoolInfo, nullptr, &cmdPool);
    if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");

    return cmdPool;
}

/**
* Create a buffer on the m_Device
*
* @param usageFlags Usage flag bit mask for the buffer (i.e. index, vertex, uniform buffer)
* @param memoryPropertyFlags Memory m_Properties for this buffer (i.e. m_Device local, host visible, coherent)
* @param size Size of the buffer in byes
* @param buffer Pointer to the buffer handle acquired by the function
* @param memory Pointer to the memory handle acquired by the function
* @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
*
* @return VK_SUCCESS if buffer handle and memory have been created and (optionally passed) data has been copied
*/
VkResult
VulkanDevice::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size,
                           VkBuffer *buffer, VkDeviceMemory *memory, void *data) const {
    // Create the buffer handle
    VkBufferCreateInfo bufferCreateInfo = Populate::bufferCreateInfo(usageFlags, size);
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_LogicalDevice, &bufferCreateInfo, nullptr, buffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Buffer");

    // Create the memory backing up the buffer handle
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(m_LogicalDevice, *buffer, &memReqs);
    assert(size <= memReqs.size);
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    memAlloc.allocationSize = memReqs.size;

    // Find a memory type index that fits the m_Properties of the buffer
    memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
    // If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to flashing the appropriate flag during allocation
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
    if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        memAlloc.pNext = &allocFlagsInfo;
    }
    if (VK_SUCCESS != vkAllocateMemory(m_LogicalDevice, &memAlloc, nullptr, memory))
        throw std::runtime_error("Failed to allocate Buffer memory");

    // If a pointer to the buffer data has been passed, map the buffer and copy over the data
    if (data != nullptr) {
        void *mapped;
        if (vkMapMemory(m_LogicalDevice, *memory, 0, size, 0, &mapped) != VK_SUCCESS)
            throw std::runtime_error("Failed to Map Buffer memory");
        memcpy(mapped, data, size);
        // If host coherency hasn't been requested, do a manual flush to make writes visible
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            VkMappedMemoryRange mappedRange = Populate::mappedMemoryRange();
            mappedRange.memory = *memory;
            mappedRange.offset = 0;
            mappedRange.size = size;
            vkFlushMappedMemoryRanges(m_LogicalDevice, 1, &mappedRange);
        }
        vkUnmapMemory(m_LogicalDevice, *memory);
    }

    // Attach the memory to the buffer object
    if (vkBindBufferMemory(m_LogicalDevice, *buffer, *memory, 0) != VK_SUCCESS)
        throw std::runtime_error("Failed to bind buffer memory");

    return VK_SUCCESS;
}

/**
* Create a buffer on the m_Device
*
* @param usageFlags Usage flag bit mask for the buffer (i.e. index, vertex, uniform buffer)
* @param memoryPropertyFlags Memory m_Properties for this buffer (i.e. m_Device local, host visible, coherent)
* @param buffer Pointer to a vk::Vulkan buffer object
* @param size Size of the buffer in bytes
* @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
*
* @return VK_SUCCESS if buffer handle and memory have been created and (optionally passed) data has been copied
*/
VkResult VulkanDevice::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags,
                                    Buffer *buffer, VkDeviceSize size, void *data) {
    buffer->m_Device = m_LogicalDevice;

    // Create the buffer handle
    VkBufferCreateInfo bufferCreateInfo = Populate::bufferCreateInfo(usageFlags, size);
    VkResult res = vkCreateBuffer(m_LogicalDevice, &bufferCreateInfo, nullptr, &buffer->m_Buffer);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create Buffer");

    // Create the memory backing up the buffer handle
    VkMemoryRequirements memReqs;
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    vkGetBufferMemoryRequirements(m_LogicalDevice, buffer->m_Buffer, &memReqs);
    memAlloc.allocationSize = memReqs.size;
    // Find a memory type index that fits the m_Properties of the buffer
    memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
    // If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to flashing the appropriate flag during allocation
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
    if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        memAlloc.pNext = &allocFlagsInfo;
    }
    res = vkAllocateMemory(m_LogicalDevice, &memAlloc, nullptr, &buffer->m_Memory);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate memory");

    buffer->alignment = memReqs.alignment;
    buffer->m_Size = size;
    buffer->usageFlags = usageFlags;
    buffer->memoryPropertyFlags = memoryPropertyFlags;

    // If a pointer to the buffer data has been passed, map the buffer and copy over the data
    if (data != nullptr) {
        if (buffer->map() != VK_SUCCESS) throw std::runtime_error("Failed to map buffer");
        memcpy(buffer->mapped, data, size);
        if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
            buffer->flush();

        buffer->unmap();
    }

    // Initialize a default m_Descriptor that covers the whole buffer size
    buffer->setupDescriptor();

    // Attach the memory to the buffer object
    return buffer->bind();
}

/**
* Copy Buffer data from src to dst using VkCmdCopyBuffer
*
* @param src Pointer to the source buffer to copy from
* @param dst Pointer to the destination buffer to copy to
* @param queue Pointer
* @param copyRegion (Optional) Pointer to a copy region, if NULL, the whole buffer is copied
*
* @note Source and destination pointers must have the appropriate transfer usage flags set (TRANSFER_SRC / TRANSFER_DST)
*/
void VulkanDevice::copyBuffer(Buffer *src, Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion) {
    assert(dst->m_Size <= src->m_Size);
    assert(src->m_Buffer);
    VkCommandBuffer copyCmd = createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy bufferCopy{};
    if (copyRegion == nullptr) {
        bufferCopy.size = src->m_Size;
    } else {
        bufferCopy = *copyRegion;
    }

    vkCmdCopyBuffer(copyCmd, src->m_Buffer, dst->m_Buffer, 1, &bufferCopy);

    flushCommandBuffer(copyCmd, queue);
}

/**
 * @brief Function to copy data between VkBuffer
 * @param src Buffer source to copy from
 * @param dst Buffer to copy to
 * @param copyRegion How much buffer to copy
 */
void VulkanDevice::copyVkBuffer(VkBuffer *src, VkBuffer *dst, VkBufferCopy *copyRegion) {

    VkCommandBuffer copyCmd = createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    vkCmdCopyBuffer(copyCmd, *src, *dst, 1, copyRegion);
    flushCommandBuffer(copyCmd, m_TransferQueue, true);

}

/**
* Allocate a command buffer from the command pool
*
* @param level Level of the new command buffer (primary or secondary)
* @param pool Command pool from which the command buffer will be allocated
* @param (Optional) begin If true, recording on the new command buffer will be started (vkBeginCommandBuffer) (Defaults to false)
*
* @return A handle to the allocated command buffer
*/
VkCommandBuffer VulkanDevice::createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin) {
    VkCommandBufferAllocateInfo cmdBufAllocateInfo = Populate::commandBufferAllocateInfo(pool, level, 1);
    VkCommandBuffer cmdBuffer;
    if (vkAllocateCommandBuffers(m_LogicalDevice, &cmdBufAllocateInfo, &cmdBuffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");
    // If requested, also start recording for the new command buffer
    if (begin) {
        VkCommandBufferBeginInfo cmdBufInfo = Populate::commandBufferBeginInfo();
        if (vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin command buffer");
    }
    return cmdBuffer;
}

VkCommandBuffer VulkanDevice::createCommandBuffer(VkCommandBufferLevel level, bool begin) {
    return createCommandBuffer(level, m_CommandPool, begin);
}

/**
* Finish command buffer recording and submit it to a queue
*
* @param commandBuffer Command buffer to flush
* @param queue Queue to submit the command buffer to
* @param pool Command pool on which the command buffer has been created
* @param free (Optional) Free the command buffer once it has been submitted (Defaults to true)
*
* @note The queue that the command buffer is submitted to must be from the same family index as the pool it was allocated from
* @note Uses a fence to ensure command buffer has finished executing
*/
void VulkanDevice::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free) {
    if (commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) throw std::runtime_error("Failed to end command buffer");

    VkSubmitInfo submitInfo = Populate::submitInfo();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
    VkFence fence;
    VkResult res = vkCreateFence(m_LogicalDevice, &fenceInfo, nullptr, &fence);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create fence");
    // Submit to the queue
    res = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to submit to queue");
    // Wait for the fence to signal that command buffer has finished executing
    res = vkWaitForFences(m_LogicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to wait for fence");
    vkDestroyFence(m_LogicalDevice, fence, nullptr);
    if (free) {
        vkFreeCommandBuffers(m_LogicalDevice, pool, 1, &commandBuffer);
    }
}

void VulkanDevice::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free) {
    return flushCommandBuffer(commandBuffer, queue, m_CommandPool, free);
}

void VulkanDevice::beginCommandBuffer(VkCommandBuffer commandBuffer) {
    VkCommandBufferBeginInfo commandBufferBI{};
    commandBufferBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &commandBufferBI));
}
