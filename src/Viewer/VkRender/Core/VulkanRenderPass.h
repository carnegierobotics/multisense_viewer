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

    /**@brief Very early iteration of a editor create info which also includes renderpass create info. TODO They should be separated into EditorCreateInfo and RenderPass even though they share a lot of information*/
    struct VulkanRenderPassCreateInfo {
        Renderer *context = nullptr;
        int32_t width = 0, appWidth = 0;
        int32_t height = 0, appHeight = 0;
        int32_t x = 0;
        int32_t y = 0;
        uint32_t borderSize = 3;
        std::string editorTypeDescription;
        bool resizeable = true;
        size_t editorIndex = 0;
        std::vector<std::string> uiLayers;
        ImGuiContext* uiContext = nullptr;

        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkFramebuffer* frameBuffers{};
        std::shared_ptr<GuiResources> guiResources;
        SharedContextData* sharedUIContextData;

        static void copy(VulkanRenderPassCreateInfo* dst, VulkanRenderPassCreateInfo* src){
            dst->appHeight = src->appHeight;
            dst->appWidth = src->appWidth;
            dst->height = src->height;
            dst->width = src->width;
            dst->x = src->x;
            dst->y = src->y;
            dst->borderSize = src->borderSize;
            dst->editorTypeDescription = src->editorTypeDescription;
            dst->resizeable = src->resizeable;
            dst->editorIndex = src->editorIndex;
            dst->uiContext = src->uiContext;
            dst->uiLayers = src->uiLayers;
        }


        VulkanRenderPassCreateInfo(VkFramebuffer * fbPtr, std::shared_ptr<GuiResources> guiRes, Renderer *ctx, SharedContextData* sharedData)
                : context(ctx), frameBuffers(fbPtr), guiResources(std::move(guiRes)), sharedUIContextData(sharedData) {

        }

    };

    struct VulkanRenderPass {
    public:

        VulkanRenderPass() = delete;

        explicit VulkanRenderPass(const VulkanRenderPassCreateInfo &createInfo);

        // Implement move constructor
        VulkanRenderPass(VulkanRenderPass &&other)  noexcept : m_logicalDevice(other.m_logicalDevice), m_allocator(other.m_allocator) {
            std::swap(this->m_colorImage, other.m_colorImage);
            std::swap(this->m_depthStencil, other.m_depthStencil);
            std::swap(this->m_renderPass, other.m_renderPass);
            std::swap(this->m_initialized, other.m_initialized);
        }
        // and move assignment operator
        VulkanRenderPass &operator=(VulkanRenderPass &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                std::swap(this->m_logicalDevice, other.m_logicalDevice);
                std::swap(this->m_allocator, other.m_allocator);
                std::swap(this->m_colorImage, other.m_colorImage);
                std::swap(this->m_depthStencil, other.m_depthStencil);
                std::swap(this->m_renderPass, other.m_renderPass);
                std::swap(this->m_initialized, other.m_initialized);
            }
            return *this;
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
        bool m_initialized = false;

        void cleanUp();
    };
}

#endif //MULTISENSE_VIEWER_VULKANRENDERPASS_H
