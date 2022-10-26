//
// Created by magnus on 9/5/21.
//


#include "Buffer.h"

/*
* Vulkan buffer class
*
* Encapsulates a Vulkan buffer
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/
/**
* Map a memory range of this buffer. If successful, mapped points to the specified buffer range.
*
* @param size (Optional) Size of the memory range to map. Pass VK_WHOLE_SIZE to map the complete buffer range.
* @param offset (Optional) Byte offset from beginning
*
* @return VkResult of the buffer mapping call
*/
VkResult Buffer::map(VkDeviceSize size, VkDeviceSize offset) {
    return vkMapMemory(m_Device, m_Memory, offset, size, 0, &mapped);
}

/**
* Unmap a mapped memory range
*
* @note Does not return a result as vkUnmapMemory can't fail
*/
void Buffer::unmap() {
    if (mapped) {
        vkUnmapMemory(m_Device, m_Memory);
        mapped = nullptr;
    }
}

/**
* Attach the allocated memory block to the buffer
*
* @param offset (Optional) Byte offset (from the beginning) for the memory region to bind
*
* @return VkResult of the bindBufferMemory call
*/
VkResult Buffer::bind(VkDeviceSize offset) {
    return vkBindBufferMemory(m_Device, m_Buffer, m_Memory, offset);
}

/**
* Setup the default m_Descriptor for this buffer
*
* @param size (Optional) Size of the memory range of the m_Descriptor
* @param offset (Optional) Byte offset from beginning
*
*/
void Buffer::setupDescriptor(VkDeviceSize size, VkDeviceSize offset) {
    m_DescriptorBufferInfo.offset = offset;
    m_DescriptorBufferInfo.buffer = m_Buffer;
    m_DescriptorBufferInfo.range = size;
}

/**
* Copies the specified data to the mapped buffer
*
* @param data Pointer to the data to copy
* @param size Size of the data to copy in machine units
*
*/
void Buffer::copyTo(void *data, VkDeviceSize size) {
    assert(mapped);
    memcpy(mapped, data, size);
}

/**
* Flush a memory range of the buffer to make it visible to the m_Device
*
* @note Only required for non-coherent memory
*
* @param size (Optional) Size of the memory range to flush. Pass VK_WHOLE_SIZE to flush the complete buffer range.
* @param offset (Optional) Byte offset from beginning
*
* @return VkResult of the flush call
*/
VkResult Buffer::flush(VkDeviceSize size, VkDeviceSize offset) {
    VkMappedMemoryRange mappedRange = {};
    mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedRange.memory = m_Memory;
    mappedRange.offset = offset;
    mappedRange.size = size;
    return vkFlushMappedMemoryRanges(m_Device, 1, &mappedRange);
}

/**
* Invalidate a memory range of the buffer to make it visible to the host
*
* @note Only required for non-coherent memory
*
* @param size (Optional) Size of the memory range to invalidate. Pass VK_WHOLE_SIZE to invalidate the complete buffer range.
* @param offset (Optional) Byte offset from beginning
*
* @return VkResult of the invalidate call
*/
VkResult Buffer::invalidate(VkDeviceSize size, VkDeviceSize offset) {
    VkMappedMemoryRange mappedRange = {};
    mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedRange.memory = m_Memory;
    mappedRange.offset = offset;
    mappedRange.size = size;
    return vkInvalidateMappedMemoryRanges(m_Device, 1, &mappedRange);
}

/**
* Release all Vulkan resources held by this buffer
*/
void Buffer::destroy() const {
    if (m_Buffer) {
        vkDestroyBuffer(m_Device, m_Buffer, nullptr);
    }
    if (m_Memory) {
        vkFreeMemory(m_Device, m_Memory, nullptr);
    }
}

