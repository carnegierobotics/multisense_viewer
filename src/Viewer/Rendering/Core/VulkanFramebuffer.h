//
// Created by magnus on 8/22/24.
//

#ifndef MULTISENSE_VIEWER_VULKANFRAMEBUFFER_H
#define MULTISENSE_VIEWER_VULKANFRAMEBUFFER_H

#include "VulkanDevice.h"

namespace VkRender {

    struct VulkanFramebufferCreateInfo {
        VulkanFramebufferCreateInfo() = delete;

        VulkanFramebufferCreateInfo(VulkanDevice &device)
                : vulkanDevice(device) {
        }

        VulkanFramebufferCreateInfo(VulkanDevice &device, std::vector<VkImageView> &attachments)
                : vulkanDevice(device), frameBufferAttachments(attachments) {
        }


        VulkanDevice &vulkanDevice;
        uint32_t height = 0;
        uint32_t width = 0;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        std::vector<VkImageView> frameBufferAttachments{};
        std::string debugInfo = "Unnamed";
        VkImageAspectFlags aspectMask{};
    };

    struct VulkanFramebuffer {
    public:

        VulkanFramebuffer() = delete;

        explicit VulkanFramebuffer(VulkanFramebufferCreateInfo &createInfo);

        // Implement move constructor
        VulkanFramebuffer(VulkanFramebuffer &&other) noexcept: m_vulkanDevice(other.m_vulkanDevice){
            std::swap(this->m_framebuffer, other.m_framebuffer);
        }

        // and move assignment operator
        VulkanFramebuffer &operator=(VulkanFramebuffer &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                std::swap(this->m_vulkanDevice, other.m_vulkanDevice);
                std::swap(this->m_framebuffer, other.m_framebuffer);
            }
            return *this;
        }

        // No copying allowed
        VulkanFramebuffer(const VulkanFramebuffer &) = delete;

        VulkanFramebuffer &operator=(const VulkanFramebuffer &) = delete;

        ~VulkanFramebuffer();

        VkFramebuffer &framebuffer() { return m_framebuffer; }


    private:
        VulkanDevice &m_vulkanDevice;
        VkFramebuffer m_framebuffer;
    };
}

#endif //MULTISENSE_VIEWER_VULKANFRAMEBUFFER_H
