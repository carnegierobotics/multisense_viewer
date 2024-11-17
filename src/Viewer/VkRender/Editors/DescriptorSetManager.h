//
// Created by magnus-desktop on 11/3/24.
//

#ifndef DESCRIPTORSETMANAGER_H
#define DESCRIPTORSETMANAGER_H
#include <functional>
#include <Viewer/VkRender/Core/VulkanDevice.h>
#include <vulkan/vulkan_core.h>


namespace VkRender{
    class DescriptorSetManager {
    public:
        DescriptorSetManager(
            VulkanDevice& device,
            const std::vector<VkDescriptorSetLayoutBinding>& bindings,
            uint32_t maxDescriptorSets = 100);
        ~DescriptorSetManager();
        VkDescriptorSetLayout& getDescriptorSetLayout(){return m_descriptorSetLayout;}

        VkDescriptorSet getOrCreateDescriptorSet(const std::vector<VkWriteDescriptorSet>& descriptorWrites);

        void freeDescriptorSets();

    private:
        VulkanDevice& m_device;
        VkDescriptorPool m_descriptorPool{};
        VkDescriptorSetLayout m_descriptorSetLayout{};
        std::unordered_map<size_t, VkDescriptorSet> m_descriptorSetCache;

        size_t hashDescriptorImageInfo(const VkDescriptorImageInfo& imageInfo);
        size_t hashDescriptorBufferInfo(const VkDescriptorBufferInfo& bufferInfo);
        size_t hashWriteDescriptorSet(const VkWriteDescriptorSet& write);
        size_t hashDescriptorWrites(const std::vector<VkWriteDescriptorSet>& writes);
    };

}



#endif //DESCRIPTORSETMANAGER_H
