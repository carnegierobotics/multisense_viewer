//
// Created by magnus on 7/30/24.
//

#ifndef MULTISENSE_VIEWER_VULKANGRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_VULKANGRAPHICSPIPELINE_H

#include <vulkan/vulkan_core.h>
#include "Viewer/Application/pch.h"
#include "VulkanDevice.h"

namespace VkRender {

    struct VulkanGraphicsPipelineCreateInfo {
        VulkanGraphicsPipelineCreateInfo() = delete;

        VulkanGraphicsPipelineCreateInfo(const VkRenderPass &pass, VulkanDevice &device)
                : renderPass(pass), vulkanDevice(device) {
        }

        VulkanGraphicsPipelineCreateInfo(const VkRenderPass &pass, VulkanDevice &device,
                                         const std::vector<VkPipelineShaderStageCreateInfo> &shaderStages)
                : renderPass(pass), vulkanDevice(device), shaders(shaderStages) {
        }

        VulkanDevice &vulkanDevice;
        std::vector<VkPipelineShaderStageCreateInfo> shaders{};
        VkRenderPass renderPass = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        size_t pushConstBlockSize = 0;
        VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineVertexInputStateCreateInfo vertexInputState;
        bool depthTesting = VK_TRUE;

        std::string debugInfo;
        VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo;
    };

    struct VulkanGraphicsPipeline {
    public:

        VulkanGraphicsPipeline() = delete;

        explicit    VulkanGraphicsPipeline(const VulkanGraphicsPipelineCreateInfo &createInfo);

        // Implement move constructor
        VulkanGraphicsPipeline(VulkanGraphicsPipeline &&other) noexcept: m_vulkanDevice(other.m_vulkanDevice) {
            std::swap(this->m_pipelineLayout, other.m_pipelineLayout);
            std::swap(this->m_pipelineCache, other.m_pipelineCache);
            std::swap(this->m_pipeline, other.m_pipeline);
            std::swap(this->m_shaderModules, other.m_shaderModules);
            std::swap(this->m_shaders, other.m_shaders);
        }

        // and move assignment operator
        VulkanGraphicsPipeline &operator=(VulkanGraphicsPipeline &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                std::swap(this->m_pipelineLayout, other.m_pipelineLayout);
                std::swap(this->m_pipelineCache, other.m_pipelineCache);
                std::swap(this->m_pipeline, other.m_pipeline);
                std::swap(this->m_shaderModules, other.m_shaderModules);
                std::swap(this->m_shaders, other.m_shaders);
                std::swap(this->m_vulkanDevice, other.m_vulkanDevice);
            }
            return *this;
        }

        // No copying allowed
        VulkanGraphicsPipeline(const VulkanGraphicsPipeline &) = delete;

        VulkanGraphicsPipeline &operator=(const VulkanGraphicsPipeline &) = delete;

        ~VulkanGraphicsPipeline();

        VkPipeline &getPipeline() {
            return m_pipeline;
        }

        VkPipelineLayout &getPipelineLayout() {
            return m_pipelineLayout;
        }

    private:
        VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
        VkPipelineCache m_pipelineCache{VK_NULL_HANDLE};
        VkPipeline m_pipeline{VK_NULL_HANDLE};
        std::vector<VkShaderModule> m_shaderModules{};
        std::vector<VkPipelineShaderStageCreateInfo> m_shaders{};
        VulkanDevice &m_vulkanDevice;
    };
}

#endif //MULTISENSE_VIEWER_VULKANGRAPHICSPIPELINE_H
