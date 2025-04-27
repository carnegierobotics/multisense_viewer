//
// Created by mgjer on 27/04/2025.
//
// VulkanShaderModule.cpp
#include "VulkanShaderModule.h"
#include <cassert>

namespace VkRender {

VulkanShaderModule::VulkanShaderModule(const VulkanShaderModuleCreateInfo& ci)
  : m_device(&ci.vulkanDevice)
  , m_debugInfo(ci.debugInfo)
{
    // 1) Create VkShaderModule
    VkShaderModuleCreateInfo mci{};
    mci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mci.codeSize = ci.spirvAsset->code.size() * sizeof(uint32_t);
    mci.pCode    = ci.spirvAsset->code.data();

    VkResult res = vkCreateShaderModule(m_device->m_LogicalDevice,
                                        &mci, nullptr, &m_module);
    assert(res == VK_SUCCESS && "Failed to create shader module");

    // 2) Fill out stage info
    m_stageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    m_stageInfo.stage  = ci.shaderStage;
    m_stageInfo.module = m_module;
    m_stageInfo.pName  = "main";
}

VulkanShaderModule::VulkanShaderModule(VulkanShaderModule&& other) noexcept {
    m_device    = other.m_device;
    m_module    = other.m_module;
    m_stageInfo = other.m_stageInfo;
    m_debugInfo = std::move(other.m_debugInfo);

    other.m_module = VK_NULL_HANDLE;
}

VulkanShaderModule& VulkanShaderModule::operator=(VulkanShaderModule&& other) noexcept {
    if (this != &other) {
        // Destroy existing
        this->~VulkanShaderModule();

        m_device    = other.m_device;
        m_module    = other.m_module;
        m_stageInfo = other.m_stageInfo;
        m_debugInfo = std::move(other.m_debugInfo);

        other.m_module = VK_NULL_HANDLE;
    }
    return *this;
}

VulkanShaderModule::~VulkanShaderModule() {
    if (m_module == VK_NULL_HANDLE)
        return;

    // Defer destruction until GPU is idle
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    vkCreateFence(m_device->m_LogicalDevice, &fci, nullptr, &fence);

    auto logicalDevice = m_device->m_LogicalDevice;
    auto moduleToDestroy = m_module;
    VulkanResourceManager::getInstance().deferDeletion(
        [logicalDevice, moduleToDestroy]() {
            vkDestroyShaderModule(logicalDevice, moduleToDestroy, nullptr);
        },
        fence,
        std::string("Deferred destroy shader module: ") + m_debugInfo
    );
}

} // namespace VkRender
