// VulkanShaderModule.h
#pragma once

#include "VulkanDevice.h"
#include "Viewer/Assets/ShaderLoader.h"
#include "VulkanResourceManager.h"
#include <vulkan/vulkan.h>
#include <string>

namespace VkRender {

    struct VulkanShaderModuleCreateInfo {
        VulkanShaderModuleCreateInfo() = delete;
        VulkanShaderModuleCreateInfo(VulkanDevice& device,
                                     std::shared_ptr<SPIRVAsset> spirv,
                                     VkShaderStageFlagBits stage,
                                     std::string debugLabel = "UnnamedShader")
          : vulkanDevice(device)
          , spirvAsset(std::move(spirv))
          , shaderStage(stage)
          , debugInfo(std::move(debugLabel))
        {}

        VulkanDevice&               vulkanDevice;
        std::shared_ptr<SPIRVAsset> spirvAsset;
        VkShaderStageFlagBits       shaderStage;
        std::string                 debugInfo;
    };

    class VulkanShaderModule {
    public:
        VulkanShaderModule() = delete;

        explicit VulkanShaderModule(const VulkanShaderModuleCreateInfo& ci);

        // Move support
        VulkanShaderModule(VulkanShaderModule&& other) noexcept;
        VulkanShaderModule& operator=(VulkanShaderModule&& other) noexcept;

        // No copying
        VulkanShaderModule(const VulkanShaderModule&) = delete;
        VulkanShaderModule& operator=(const VulkanShaderModule&) = delete;

        ~VulkanShaderModule();

        VkShaderModule                               module() const { return m_module; }
        const VkPipelineShaderStageCreateInfo&       stageInfo() const { return m_stageInfo; }

    private:
        VulkanDevice*                 m_device      = nullptr;
        VkShaderModule                m_module      = VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo m_stageInfo = {};
        std::string                   m_debugInfo;
    };

} // namespace VkRender
