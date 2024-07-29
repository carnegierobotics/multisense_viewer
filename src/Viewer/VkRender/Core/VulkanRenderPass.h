//
// Created by magnus on 7/17/24.
//

#ifndef MULTISENSE_VIEWER_VULKANRENDERPASS_H
#define MULTISENSE_VIEWER_VULKANRENDERPASS_H

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>
#include <string>
#include <memory>
#include <utility>

#include "Viewer/VkRender/ImGui/GuiManager.h"

namespace VkRender {
    class Renderer;

    struct VulkanRenderPassCreateInfo {
        Renderer *context = nullptr;
        int32_t width = 0, appWidth = 0;
        int32_t height = 0, appHeight = 0;
        int32_t x = 0;
        int32_t y = 0;
        uint32_t borderSize = 3;
        std::string editorTypeDescription;
        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkFramebuffer* frameBuffers;
        std::shared_ptr<GuiResources> guiResources;
        std::vector<VkClearValue> clearValue;
        bool resizeable = true;

        VulkanRenderPassCreateInfo() = default;

        VulkanRenderPassCreateInfo(VkFramebuffer * fbPtr, std::shared_ptr<GuiResources> guiRes, Renderer *ctx)
                : context(ctx), frameBuffers(fbPtr), guiResources(std::move(guiRes)) {

        }

    };

    struct VulkanRenderPass {
    public:

        explicit VulkanRenderPass(const VulkanRenderPassCreateInfo &createInfo);

        // Implement move constructor
        VulkanRenderPass(VulkanRenderPass &&other)  noexcept : m_logicalDevice(other.m_logicalDevice), m_allocator(other.m_allocator) {
            swap(*this, other);
        }

        // and move assignment operator
        VulkanRenderPass &operator=(VulkanRenderPass &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                swap(*this, other);
            }
            return *this;
        }
        // Implement a swap function
        friend void swap(VulkanRenderPass &first, VulkanRenderPass &second) noexcept {
            std::swap(first.m_logicalDevice, second.m_logicalDevice);
            std::swap(first.m_allocator, second.m_allocator);
            std::swap(first.m_colorImage, second.m_colorImage);
            std::swap(first.m_depthStencil, second.m_depthStencil);
            std::swap(first.m_renderPass, second.m_renderPass);
        }

        // No copying allowed
        VulkanRenderPass(const VulkanRenderPass &) = delete;
        VulkanRenderPass &operator=(const VulkanRenderPass &) = delete;

        ~VulkanRenderPass();

        VkRenderPass &getRenderPass() {
            return m_renderPass;
        }

    private:
        VkRenderPass m_renderPass = VK_NULL_HANDLE;
        struct {
            VkImage image = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
            VmaAllocation allocation{};
        } m_depthStencil{};
        struct {
            VkImage image = VK_NULL_HANDLE;
            VkImage resolvedImage = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            VkDeviceMemory resolvedMem = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkImageView resolvedView = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
            VmaAllocation colorImageAllocation{};
            VmaAllocation resolvedImageAllocation{};
        } m_colorImage{};

        VkDevice &m_logicalDevice;
        VmaAllocator &m_allocator;

        void cleanUp();
    };
}

#endif //MULTISENSE_VIEWER_VULKANRENDERPASS_H
