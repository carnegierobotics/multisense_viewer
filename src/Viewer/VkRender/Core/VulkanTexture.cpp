//
// Created by mgjer on 24/08/2024.
//

#include "VulkanTexture.h"

namespace VkRender {


    VulkanTexture::VulkanTexture(VulkanTexture2DCreateInfo &createInfo) : m_vulkanDevice(createInfo.vulkanDevice){
        m_image = createInfo.image;
        // Create m_Sampler
        VkSamplerCreateInfo samplerCreateInfo = {};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;

        CHECK_RESULT(vkCreateSampler(m_vulkanDevice.m_LogicalDevice, &samplerCreateInfo, nullptr, &m_sampler))


        m_imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_imageInfo.imageView = m_image->view(); // Your depth image view
        m_imageInfo.sampler = m_sampler; // Sampler you've created for depth texture
    }

    VulkanTexture::~VulkanTexture() {
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(m_vulkanDevice.m_LogicalDevice, &fenceCreateInfo, nullptr, &fence);
        // Capture all necessary members by value
        auto sampler = m_sampler;
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, sampler]() {
                    vkDestroySampler(logicalDevice, sampler, nullptr);
                },
                fence);
    }


}