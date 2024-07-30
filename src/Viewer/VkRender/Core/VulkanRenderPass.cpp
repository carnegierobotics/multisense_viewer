//
// Created by magnus on 7/17/24.
//

#include <array>
#include "VulkanRenderPass.h"

#include "Viewer/Tools/Populate.h"
#include "Viewer/VkRender/Editor.h"
#include "Viewer/VkRender/Renderer.h"
#include "VulkanResourceManager.h"

namespace VkRender {


    VulkanRenderPass::VulkanRenderPass(const VulkanRenderPassCreateInfo &createInfo) : m_logicalDevice(
            createInfo.context->vkDevice()), m_allocator(createInfo.context->allocator()) {

        VkSampleCountFlagBits sampleCount = createInfo.context->data().msaaSamples;
        uint32_t width =  std::max(createInfo.width , 0);
        uint32_t height = std::max(createInfo.height, 0);
        const auto &data = createInfo.context->data();
        std::string editorTypeDescription = createInfo.editorTypeDescription;

        //// COLOR IMAGE RESOURCE /////
        VkImageCreateInfo colorImageCI = Populate::imageCreateInfo();
        colorImageCI.imageType = VK_IMAGE_TYPE_2D;
        colorImageCI.format = data.swapchainColorFormat;
        colorImageCI.extent = {width, height, 1};
        colorImageCI.mipLevels = 1;
        colorImageCI.arrayLayers = 1;
        colorImageCI.samples = sampleCount;
        colorImageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorImageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        colorImageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        // Create the color image using VMA
        VmaAllocationCreateInfo allocCreateInfo = {};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkResult result = vmaCreateImage(createInfo.context->allocator(), &colorImageCI, &allocCreateInfo,
                                         &m_colorImage.image,
                                         &m_colorImage.colorImageAllocation, nullptr);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create colorImage");

        VALIDATION_DEBUG_NAME(data.device->m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_colorImage.image), VK_OBJECT_TYPE_IMAGE,
                              (editorTypeDescription + "UIPassImage").c_str());

        /*
        if (multisampled) {
            // Create an additional resolved image if MSAA is used
            VkImageCreateInfo resolvedImageCI = colorImageCI; // Copy from colorImageCI for basic settings
            resolvedImageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            resolvedImageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            result = vmaCreateImage(m_context->allocator(), &resolvedImageCI, &allocCreateInfo,
                                    &colorImage.resolvedImage,
                                    &colorImage.resolvedImageAllocation, nullptr);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create resolvedImage");

            VALIDATION_DEBUG_NAME(m_renderUtils.device->m_LogicalDevice,
                                  reinterpret_cast<uint64_t>(colorImage.resolvedImage),
                                  VK_OBJECT_TYPE_IMAGE,
                                  (editorTypeDescription + "UIPassResolvedImage").c_str());
        }
        */

        VkImageViewCreateInfo colorImageViewCI = Populate::imageViewCreateInfo();
        colorImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorImageViewCI.image = m_colorImage.image;
        colorImageViewCI.format = data.swapchainColorFormat;
        colorImageViewCI.subresourceRange.baseMipLevel = 0;
        colorImageViewCI.subresourceRange.levelCount = 1;
        colorImageViewCI.subresourceRange.baseArrayLayer = 0;
        colorImageViewCI.subresourceRange.layerCount = 1;
        colorImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (data.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            colorImageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }

        result = vkCreateImageView(data.device->m_LogicalDevice, &colorImageViewCI, nullptr,
                                   &m_colorImage.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create colorImage view");

        /*
        if (multisampled) {
            colorImageViewCI.image = colorImage.resolvedImage;
            result = vkCreateImageView(data.device->m_LogicalDevice, &colorImageViewCI, nullptr,
                                       &colorImage.resolvedView);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create resolvedView");
        }
        */

        //// SAMPLER SETUP ////
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR; // Magnification filter
        samplerInfo.minFilter = VK_FILTER_LINEAR; // Minification filter
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // Wrap mode for texture coordinate U
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // Wrap mode for texture coordinate V
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; // Wrap mode for texture coordinate W
        samplerInfo.anisotropyEnable = VK_TRUE; // Enable anisotropic filtering
        samplerInfo.maxAnisotropy = 16; // Max level of anisotropy
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK; // Border color when using VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
        samplerInfo.unnormalizedCoordinates = VK_FALSE; // Use normalized texture coordinates
        samplerInfo.compareEnable = VK_FALSE; // Enable comparison mode for the sampler
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS; // Comparison operator if compareEnable is VK_TRUE
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; // Mipmap interpolation mode
        samplerInfo.minLod = 0; // Minimum level of detail
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE; // Maximum level of detail
        samplerInfo.mipLodBias = 0.0f; // Level of detail bias

        if (vkCreateSampler(data.device->m_LogicalDevice, &samplerInfo, nullptr,
                            &m_colorImage.sampler) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

        VkAttachmentDescription uiColorAttachment = {};
        uiColorAttachment.format = data.swapchainColorFormat;
        uiColorAttachment.samples = data.msaaSamples;
        uiColorAttachment.loadOp = createInfo.loadOp; // Load since it follows the main pass
        uiColorAttachment.storeOp = createInfo.storeOp;
        uiColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiColorAttachment.initialLayout = createInfo.initialLayout;
        uiColorAttachment.finalLayout = createInfo.finalLayout;
        VkAttachmentReference uiColorAttachmentRef = {};
        uiColorAttachmentRef.attachment = 0;
        uiColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription uiResolveAttachment = {};
        uiResolveAttachment.format = data.swapchainColorFormat;
        uiResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        uiResolveAttachment.loadOp = createInfo.loadOp;
        uiResolveAttachment.storeOp = createInfo.storeOp;
        uiResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiResolveAttachment.initialLayout = createInfo.initialLayout;
        uiResolveAttachment.finalLayout = createInfo.finalLayout;
        VkAttachmentReference uiResolveAttachmentRef = {};
        uiResolveAttachmentRef.attachment = 2;
        uiResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription dummyDepthAttachment = {};
        dummyDepthAttachment.format = data.depthFormat;
        dummyDepthAttachment.samples = data.msaaSamples;
        dummyDepthAttachment.loadOp = createInfo.loadOp;
        dummyDepthAttachment.storeOp = createInfo.storeOp;
        dummyDepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        dummyDepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        dummyDepthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        dummyDepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference dummyDepthAttachmentRef = {};
        dummyDepthAttachmentRef.attachment = 1;
        dummyDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription uiSubpass = {};
        uiSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        uiSubpass.colorAttachmentCount = 1;
        uiSubpass.pColorAttachments = &uiColorAttachmentRef;
        uiSubpass.pResolveAttachments = &uiResolveAttachmentRef;
        uiSubpass.pDepthStencilAttachment = &dummyDepthAttachmentRef;

        std::array<VkSubpassDependency, 2> uiDependencies = {};
        uiDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        uiDependencies[0].dstSubpass = 0;
        uiDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        uiDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        uiDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        uiDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        uiDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        uiDependencies[1].srcSubpass = 0;
        uiDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        uiDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        uiDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        uiDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        uiDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        uiDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        std::array<VkAttachmentDescription, 3> uiAttachments = {uiColorAttachment, dummyDepthAttachment,
                                                                uiResolveAttachment};
        VkRenderPassCreateInfo uiRenderPassInfo = {};
        uiRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        uiRenderPassInfo.attachmentCount = static_cast<uint32_t>(uiAttachments.size());
        uiRenderPassInfo.pAttachments = uiAttachments.data();
        uiRenderPassInfo.subpassCount = 1;
        uiRenderPassInfo.pSubpasses = &uiSubpass;
        uiRenderPassInfo.dependencyCount = static_cast<uint32_t>(uiDependencies.size());
        uiRenderPassInfo.pDependencies = uiDependencies.data();

        if (vkCreateRenderPass(data.device->m_LogicalDevice, &uiRenderPassInfo, nullptr,
                               &m_renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create UI render pass!");
        }
        VALIDATION_DEBUG_NAME(data.device->m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_renderPass), VK_OBJECT_TYPE_RENDER_PASS,
                              (editorTypeDescription + "UIRenderPass").c_str());

        m_initialized = true;
    }

    void VulkanRenderPass::cleanUp() {
        vkDestroyRenderPass(m_logicalDevice, m_renderPass, nullptr);
        vkDestroySampler(m_logicalDevice, m_colorImage.sampler, nullptr);

        // Destroy the image view
        if (m_colorImage.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_logicalDevice, m_colorImage.view, nullptr);
        }

// Destroy the image and free its memory allocation using VMA
        if (m_colorImage.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_allocator, m_colorImage.image,
                            m_colorImage.colorImageAllocation);
        }
// Destroy the image and free its memory allocation using VMA
        if (m_depthStencil.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_allocator, m_depthStencil.image,
                            m_depthStencil.allocation);
        }

        // Destroy the resolved image view
        if (m_depthStencil.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_logicalDevice, m_depthStencil.view,
                               nullptr);
        }
    }

    VulkanRenderPass::~VulkanRenderPass() {
        if (m_initialized) {
            VkFence fence;
            VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
            vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &fence);

            // Capture all necessary members by value
            auto logicalDevice = m_logicalDevice;
            auto allocator = m_allocator;
            auto colorImage = m_colorImage;
            auto depthStencil = m_depthStencil;
            auto renderPass = m_renderPass;

            VulkanResourceManager::getInstance().deferDeletion(
                    [logicalDevice, allocator, colorImage, depthStencil, renderPass]() {
                        // Cleanup logic with captured values
                        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
                        vkDestroySampler(logicalDevice, colorImage.sampler, nullptr);

                        if (colorImage.view != VK_NULL_HANDLE) {
                            vkDestroyImageView(logicalDevice, colorImage.view, nullptr);
                        }

                        if (colorImage.image != VK_NULL_HANDLE) {
                            vmaDestroyImage(allocator, colorImage.image, colorImage.colorImageAllocation);
                        }

                        if (depthStencil.image != VK_NULL_HANDLE) {
                            vmaDestroyImage(allocator, depthStencil.image, depthStencil.allocation);
                        }

                        if (depthStencil.view != VK_NULL_HANDLE) {
                            vkDestroyImageView(logicalDevice, depthStencil.view, nullptr);
                        }
                    },
                    fence);

        }

    }
}