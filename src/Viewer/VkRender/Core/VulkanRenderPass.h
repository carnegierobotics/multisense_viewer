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
        uint32_t width = 0, appWidth = 0;
        uint32_t height = 0, appHeight = 0;
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t borderSize = 3;
        std::string editorTypeDescription;
        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkImageView colorImageView = VK_NULL_HANDLE;
        VkImageView depthImageView = VK_NULL_HANDLE;
        std::shared_ptr<GuiResources> guiResources;
        std::vector<VkClearValue> clearValue;
        bool resizeable = true;

        VulkanRenderPassCreateInfo() = default;

        VulkanRenderPassCreateInfo(VkImageView &colorView,
                                   VkImageView &depthView, std::shared_ptr<GuiResources> guiRes, Renderer *ctx)
                : context(ctx), colorImageView(colorView), depthImageView(depthView), guiResources(std::move(guiRes)) {

        }

    };

    struct VulkanRenderPass {
    public:
        VulkanRenderPass() = delete;

        explicit VulkanRenderPass(const VulkanRenderPassCreateInfo &createInfo);

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
        VkDescriptorImageInfo m_imageInfo{};

        VkDevice &m_logicalDevice;
        VmaAllocator &m_allocator;

    };
}

#endif //MULTISENSE_VIEWER_VULKANRENDERPASS_H
