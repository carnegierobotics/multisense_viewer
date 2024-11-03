//
// Created by magnus-desktop on 11/3/24.
//

#include "DescriptorSetManager.h"

namespace VkRender {
    DescriptorSetManager::DescriptorSetManager(
    VulkanDevice& device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    const uint32_t maxDescriptorSets)
    : m_device(device)
    {
        // Create the descriptor set layout
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        VkResult result = vkCreateDescriptorSetLayout(m_device.m_LogicalDevice, &layoutInfo, nullptr, &m_descriptorSetLayout);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        // Calculate descriptor pool sizes based on the provided bindings
        std::unordered_map<VkDescriptorType, uint32_t> descriptorTypeCounts;
        for (const auto& binding : bindings) {
            descriptorTypeCounts[binding.descriptorType] += binding.descriptorCount * maxDescriptorSets;
        }

        std::vector<VkDescriptorPoolSize> poolSizes;
        for (const auto& [type, count] : descriptorTypeCounts) {
            VkDescriptorPoolSize poolSize{};
            poolSize.type = type;
            poolSize.descriptorCount = count;
            poolSizes.push_back(poolSize);
        }

        // Create the descriptor pool
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = maxDescriptorSets;
        poolInfo.flags = 0; // Optional flags, e.g., VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT

        result = vkCreateDescriptorPool(m_device.m_LogicalDevice, &poolInfo, nullptr, &m_descriptorPool);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }
    }

    DescriptorSetManager::~DescriptorSetManager() {
        // Destroy descriptor sets if needed (not usually necessary as they are freed with the pool)
        // Clean up the descriptor pool and set layout
        if (m_descriptorPool) {
            vkDestroyDescriptorPool(m_device.m_LogicalDevice, m_descriptorPool, nullptr);
        }
        if (m_descriptorSetLayout) {
            vkDestroyDescriptorSetLayout(m_device.m_LogicalDevice, m_descriptorSetLayout, nullptr);
        }
    }


    VkDescriptorSet DescriptorSetManager::getOrCreateDescriptorSet(const std::vector<VkWriteDescriptorSet>& descriptorWrites) {
        // Generate a unique key based on the descriptor writes (you may need a custom hash function)
        size_t hashKey = hashDescriptorWrites(descriptorWrites);

        auto it = m_descriptorSetCache.find(hashKey);
        if (it != m_descriptorSetCache.end()) {
            return it->second;
        }

        // Allocate a new descriptor set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_descriptorSetLayout;

        VkDescriptorSet descriptorSet;
        VkResult result = vkAllocateDescriptorSets(m_device.m_LogicalDevice, &allocInfo, &descriptorSet);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        // Update the descriptor set with the provided writes
        std::vector<VkWriteDescriptorSet> writes = descriptorWrites;
        for (auto& write : writes) {
            write.dstSet = descriptorSet;
        }

        vkUpdateDescriptorSets(m_device.m_LogicalDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // Cache the descriptor set
        m_descriptorSetCache[hashKey] = descriptorSet;

        return descriptorSet;
    }


    size_t DescriptorSetManager::hashDescriptorImageInfo(const VkDescriptorImageInfo& imageInfo) {
        size_t hash = 0;
        // Hash the image view
        hash ^= std::hash<VkImageView>()(imageInfo.imageView) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        // Hash the sampler
        hash ^= std::hash<VkSampler>()(imageInfo.sampler) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        // Hash the image layout
        hash ^= std::hash<VkImageLayout>()(imageInfo.imageLayout) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
    size_t DescriptorSetManager::hashDescriptorBufferInfo(const VkDescriptorBufferInfo& bufferInfo) {
        size_t hash = 0;
        // Hash the buffer handle
        hash ^= std::hash<VkBuffer>()(bufferInfo.buffer) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        // Hash the offset
        hash ^= std::hash<VkDeviceSize>()(bufferInfo.offset) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        // Hash the range
        hash ^= std::hash<VkDeviceSize>()(bufferInfo.range) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }

    size_t DescriptorSetManager::hashWriteDescriptorSet(const VkWriteDescriptorSet& write) {
        size_t hash = 0;
        // Hash the destination binding
        hash ^= std::hash<uint32_t>()(write.dstBinding) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        // Hash the descriptor type
        hash ^= std::hash<uint32_t>()(write.descriptorType) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        // Hash the descriptor count
        hash ^= std::hash<uint32_t>()(write.descriptorCount) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

        // Hash the resources
        if (write.pImageInfo) {
            for (uint32_t i = 0; i < write.descriptorCount; ++i) {
                hash ^= hashDescriptorImageInfo(write.pImageInfo[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        if (write.pBufferInfo) {
            for (uint32_t i = 0; i < write.descriptorCount; ++i) {
                hash ^= hashDescriptorBufferInfo(write.pBufferInfo[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        // Handle other descriptor types if necessary (e.g., pTexelBufferView)
        return hash;
    }

    size_t DescriptorSetManager::hashDescriptorWrites(const std::vector<VkWriteDescriptorSet>& writes) {
        size_t hash = 0;
        for (const auto& write : writes) {
            hash ^= hashWriteDescriptorSet(write) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
}
