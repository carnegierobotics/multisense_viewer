//
// Created by magnus on 7/15/24.
//


#include "Viewer/VkRender/Editor.h"
#include "Viewer/Tools/Utils.h"

#include "Viewer/VkRender/Renderer.h"

namespace VkRender {


    Editor::Editor(RenderUtils &utils, Renderer &ctx) : m_renderUtils(utils), m_context(ctx) {

        width = m_renderUtils.width / 2;
        height = m_renderUtils.height;

        createColorResources();
        setupDepthStencil();

        // Setup renderpasses
        depthRenderPass.type = "depth";
        depthRenderPass.multisampled = false;
        setupRenderPasses(&depthRenderPass);
        uiRenderPass.type = "ui";
        uiRenderPass.multisampled = true;
        setupUIRenderPass(&uiRenderPass);
        objectRenderPass.multisampled = true; // if we reuse s
        setupRenderPasses(&objectRenderPass);


        // Setup Framebuffer
        setupMainFramebuffer();




        m_guiManager = std::make_unique<GuiManager>(m_renderUtils.device,
                                                    uiRenderPass.renderPass,
                                                    width,
                                                    height,
                                                    m_renderUtils.msaaSamples,
                                                    m_renderUtils.swapchainImages,
                                                    &m_context
        );
        //m_guiManager->handles.mouse = &mouseButtons;
        //m_guiManager->handles.usageMonitor = m_usageMonitor;
        //m_guiManager->handles.m_cameraSelection.info[m_selectedCameraTag].type = m_cameras[m_selectedCameraTag].m_type;
    }

    void Editor::setupDepthStencil() {
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = m_renderUtils.depthFormat;
        imageCI.extent = {width, height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = m_renderUtils.msaaSamples;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkResult result = vmaCreateImage(m_context.allocator(), &imageCI, &allocInfo, &m_depthStencil.image, &m_depthStencil.allocation, nullptr);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image");
        vmaSetAllocationUserData(m_context.allocator(), m_depthStencil.allocation, &description);

        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = m_depthStencil.image;
        imageViewCI.format = m_renderUtils.depthFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (m_renderUtils.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &imageViewCI, nullptr, &m_depthStencil.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth image view");
    }


    void Editor::createColorResources() {
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = m_renderUtils.swapchainColorFormat;
        imageCI.extent = {width, height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = m_renderUtils.msaaSamples;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult result = vkCreateImage(m_renderUtils.device->m_LogicalDevice, &imageCI, nullptr, &m_colorImage.image);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image");

        VkMemoryRequirements memReqs{};
        vkGetImageMemoryRequirements(m_renderUtils.device->m_LogicalDevice, m_colorImage.image, &memReqs);

        VkMemoryAllocateInfo memAllloc = Populate::memoryAllocateInfo();
        memAllloc.allocationSize = memReqs.size;
        memAllloc.memoryTypeIndex = m_renderUtils.device->getMemoryType(memReqs.memoryTypeBits,
                                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(m_renderUtils.device->m_LogicalDevice, &memAllloc, nullptr, &m_colorImage.mem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
        result = vkBindImageMemory(m_renderUtils.device->m_LogicalDevice, m_colorImage.image, m_colorImage.mem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.image = m_colorImage.image;
        imageViewCI.format = m_renderUtils.swapchainColorFormat;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (m_renderUtils.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &imageViewCI, nullptr, &m_colorImage.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image m_View");
    }

    void Editor::setupMainFramebuffer() {

        // Depth/Stencil attachment is the same for all frame buffers
        if (m_renderUtils.msaaSamples == VK_SAMPLE_COUNT_1_BIT) {
            std::array<VkImageView, 2> attachments{};
            attachments[1] = m_depthStencil.view;
            VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(width, height,
                                                                                            attachments.data(),
                                                                                            attachments.size(),
                                                                                            objectRenderPass.renderPass);
            frameBuffers.resize(m_renderUtils.swapchainImages);
            for (uint32_t i = 0; i < frameBuffers.size(); i++) {
                attachments[0] = m_context.swapChainBuffers()[i].view;
                VkResult result = vkCreateFramebuffer(m_renderUtils.device->m_LogicalDevice, &frameBufferCreateInfo, nullptr, &frameBuffers[i]);
                if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
            }
        } else {
            std::array<VkImageView, 3> attachments{};
            attachments[0] = m_colorImage.view;
            attachments[1] = m_depthStencil.view;
            VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(width, height,
                                                                                            attachments.data(),
                                                                                            attachments.size(),
                                                                                            objectRenderPass.renderPass);
            frameBuffers.resize(m_renderUtils.swapchainImages);
            for (uint32_t i = 0; i < frameBuffers.size(); i++) {
                attachments[2] = m_context.swapChainBuffers()[i].view;
                VkResult result = vkCreateFramebuffer(m_renderUtils.device->m_LogicalDevice, &frameBufferCreateInfo, nullptr, &frameBuffers[i]);
                if (result != VK_SUCCESS) throw std::runtime_error("Failed to create framebuffer");
            }
        }

    }


    void Editor::setupRenderPasses(EditorRenderPass *secondaryRenderPasses) {
        // Color image resource
        // Depth stencil resource
        // Render Pass
        // Frame Buffer
        VkSampleCountFlagBits sampleCount = secondaryRenderPasses->multisampled ? m_renderUtils.msaaSamples
                                                                                : VK_SAMPLE_COUNT_1_BIT;

        //// DEPTH STENCIL RESOURCE /////
        VkImageCreateInfo depthImageCI = Populate::imageCreateInfo();
        depthImageCI.imageType = VK_IMAGE_TYPE_2D;
        depthImageCI.format = m_renderUtils.depthFormat;
        depthImageCI.extent = {m_renderUtils.width, m_renderUtils.height, 1};
        depthImageCI.mipLevels = 1;
        depthImageCI.arrayLayers = 1;
        depthImageCI.samples = sampleCount;
        depthImageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthImageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT;
        depthImageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (!secondaryRenderPasses->multisampled) {
            depthImageCI.usage |= VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // Add sampled bit if not using MSAA, useful for certain post-processing effects
        }

        VkResult result = vkCreateImage(m_renderUtils.device->m_LogicalDevice, &depthImageCI, nullptr,
                                        &secondaryRenderPasses->depthStencil.image);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image");

        VkMemoryRequirements depthMemReqs{};
        vkGetImageMemoryRequirements(m_renderUtils.device->m_LogicalDevice, secondaryRenderPasses->depthStencil.image,
                                     &depthMemReqs);

        VkMemoryAllocateInfo depthMemAllloc = Populate::memoryAllocateInfo();
        depthMemAllloc.allocationSize = depthMemReqs.size;
        depthMemAllloc.memoryTypeIndex = m_renderUtils.device->getMemoryType(depthMemReqs.memoryTypeBits,
                                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        result = vkAllocateMemory(m_renderUtils.device->m_LogicalDevice, &depthMemAllloc, nullptr,
                                  &secondaryRenderPasses->depthStencil.mem);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
        result = vkBindImageMemory(m_renderUtils.device->m_LogicalDevice, secondaryRenderPasses->depthStencil.image,
                                   secondaryRenderPasses->depthStencil.mem, 0);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        VkImageViewCreateInfo depthImageViewCI = Populate::imageViewCreateInfo();
        depthImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthImageViewCI.image = secondaryRenderPasses->depthStencil.image;
        depthImageViewCI.format = m_renderUtils.depthFormat;
        depthImageViewCI.subresourceRange.baseMipLevel = 0;
        depthImageViewCI.subresourceRange.levelCount = 1;
        depthImageViewCI.subresourceRange.baseArrayLayer = 0;
        depthImageViewCI.subresourceRange.layerCount = 1;
        depthImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (m_renderUtils.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            depthImageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &depthImageViewCI, nullptr,
                                   &secondaryRenderPasses->depthStencil.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create depth m_Image m_View");

        //// COLOR IMAGE RESOURCE /////
        VkImageCreateInfo colorImageCI = Populate::imageCreateInfo();
        colorImageCI.imageType = VK_IMAGE_TYPE_2D;
        colorImageCI.format = secondaryRenderPasses->setupFrameBuffer ? VK_FORMAT_R8G8B8A8_UNORM
                                                                      : m_renderUtils.swapchainColorFormat;
        colorImageCI.extent = {m_renderUtils.width, m_renderUtils.height, 1};
        colorImageCI.mipLevels = 1;
        colorImageCI.arrayLayers = 1;
        colorImageCI.samples = sampleCount;
        colorImageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorImageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        colorImageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        result = vkCreateImage(m_renderUtils.device->m_LogicalDevice, &colorImageCI, nullptr,
                               &secondaryRenderPasses->colorImage.image);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create colorImage");

        if (secondaryRenderPasses->multisampled) {
            // Create an additional resolved image if MSAA is used
            VkImageCreateInfo resolvedImageCI = colorImageCI; // Copy from colorImageCI for basic settings
            resolvedImageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            resolvedImageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            result = vkCreateImage(m_renderUtils.device->m_LogicalDevice, &resolvedImageCI, nullptr,
                                   &secondaryRenderPasses->colorImage.resolvedImage);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create resolvedImage");
        }


        {
            VkMemoryRequirements colorMemReqs{};
            vkGetImageMemoryRequirements(m_renderUtils.device->m_LogicalDevice, secondaryRenderPasses->colorImage.image,
                                         &colorMemReqs);

            VkMemoryAllocateInfo colorMemAllloc = Populate::memoryAllocateInfo();
            colorMemAllloc.allocationSize = colorMemReqs.size;
            colorMemAllloc.memoryTypeIndex = m_renderUtils.device->getMemoryType(colorMemReqs.memoryTypeBits,
                                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            result = vkAllocateMemory(m_renderUtils.device->m_LogicalDevice, &colorMemAllloc, nullptr,
                                      &secondaryRenderPasses->colorImage.mem);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
            result = vkBindImageMemory(m_renderUtils.device->m_LogicalDevice, secondaryRenderPasses->colorImage.image,
                                       secondaryRenderPasses->colorImage.mem, 0);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        }
        if (secondaryRenderPasses->multisampled) {
            VkMemoryRequirements colorMemReqs{};
            vkGetImageMemoryRequirements(m_renderUtils.device->m_LogicalDevice,
                                         secondaryRenderPasses->colorImage.resolvedImage, &colorMemReqs);

            VkMemoryAllocateInfo colorMemAllloc = Populate::memoryAllocateInfo();
            colorMemAllloc.allocationSize = colorMemReqs.size;
            colorMemAllloc.memoryTypeIndex = m_renderUtils.device->getMemoryType(colorMemReqs.memoryTypeBits,
                                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            result = vkAllocateMemory(m_renderUtils.device->m_LogicalDevice, &colorMemAllloc, nullptr,
                                      &secondaryRenderPasses->colorImage.resolvedMem);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
            result = vkBindImageMemory(m_renderUtils.device->m_LogicalDevice,
                                       secondaryRenderPasses->colorImage.resolvedImage,
                                       secondaryRenderPasses->colorImage.resolvedMem, 0);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");
        }


        VkImageViewCreateInfo colorImageViewCI = Populate::imageViewCreateInfo();
        colorImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorImageViewCI.image = secondaryRenderPasses->colorImage.image;
        colorImageViewCI.format = secondaryRenderPasses->setupFrameBuffer ? VK_FORMAT_R8G8B8A8_UNORM
                                                                          : m_renderUtils.swapchainColorFormat;;
        colorImageViewCI.subresourceRange.baseMipLevel = 0;
        colorImageViewCI.subresourceRange.levelCount = 1;
        colorImageViewCI.subresourceRange.baseArrayLayer = 0;
        colorImageViewCI.subresourceRange.layerCount = 1;
        colorImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (m_renderUtils.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            colorImageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }

        result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &colorImageViewCI, nullptr,
                                   &secondaryRenderPasses->colorImage.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create colorImage view");

        if (secondaryRenderPasses->multisampled) {
            colorImageViewCI.image = secondaryRenderPasses->colorImage.resolvedImage;
            result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &colorImageViewCI, nullptr,
                                       &secondaryRenderPasses->colorImage.resolvedView);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create resolvedVie");
        }
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

        if (vkCreateSampler(m_renderUtils.device->m_LogicalDevice, &samplerInfo, nullptr,
                            &secondaryRenderPasses->colorImage.sampler) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
        if (vkCreateSampler(m_renderUtils.device->m_LogicalDevice, &samplerInfo, nullptr,
                            &secondaryRenderPasses->depthStencil.sampler) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

        //// RenderPass setup ////
        std::vector<VkAttachmentDescription> attachments;
        VkSubpassDescription subpassDescription{};


        VkAttachmentReference colorReference{};
        VkAttachmentReference depthReference{};
        VkAttachmentReference colorAttachmentResolveRef{};

        VkAttachmentDescription colorAttachment{};
        // Color attachment
        colorAttachment.format = secondaryRenderPasses->setupFrameBuffer ? VK_FORMAT_R8G8B8A8_UNORM
                                                                         : m_renderUtils.swapchainColorFormat;
        colorAttachment.samples = sampleCount;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.loadOp = secondaryRenderPasses->setupFrameBuffer ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                                         : VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = secondaryRenderPasses->setupFrameBuffer ? VK_IMAGE_LAYOUT_UNDEFINED
                                                                                : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Depth attachment
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = m_renderUtils.depthFormat;
        depthAttachment.samples = sampleCount;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.initialLayout = secondaryRenderPasses->setupFrameBuffer ? VK_IMAGE_LAYOUT_UNDEFINED
                                                                                : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format = secondaryRenderPasses->setupFrameBuffer ? VK_FORMAT_R8G8B8A8_UNORM
                                                                                : m_renderUtils.swapchainColorFormat;
        colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp = secondaryRenderPasses->setupFrameBuffer ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                                                : VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout = secondaryRenderPasses->setupFrameBuffer ? VK_IMAGE_LAYOUT_UNDEFINED
                                                                                       : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachmentResolve.finalLayout = secondaryRenderPasses->setupFrameBuffer
                                             ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                             : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;  // Color attachment
        subpassDescription.pDepthStencilAttachment = &depthReference; // Depth attachment
        subpassDescription.inputAttachmentCount = 0;
        subpassDescription.pInputAttachments = nullptr;
        subpassDescription.preserveAttachmentCount = 0;
        subpassDescription.pPreserveAttachments = nullptr;

        colorReference.attachment = 0;
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        depthReference.attachment = 1;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        if (secondaryRenderPasses->multisampled) {
            attachments = {{colorAttachment, depthAttachment, colorAttachmentResolve}};

            colorAttachmentResolveRef.attachment = 2;
            colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            subpassDescription.pResolveAttachments = &colorAttachmentResolveRef; // Where to resolve the multisampled color attachment

        } else {
            attachments = {{colorAttachment, depthAttachment}};
        }


        std::array<VkSubpassDependency, 2> dependencies{};

        if (secondaryRenderPasses->setupFrameBuffer) {
            // Subpass dependencies for layout transitions
            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // Adjusted
            dependencies[0].dstStageMask =
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;;
            dependencies[0].srcAccessMask = VK_ACCESS_NONE_KHR; // Adjusted to reflect completion of writes
            dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask =
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; // Adjusted if necessary
            dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;;
            dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT; // Adjusted if subsequent operations are general
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        } else {
            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        }

        VkRenderPassCreateInfo renderPassInfo = Populate::renderPassCreateInfo();
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();


        if (vkCreateRenderPass(m_renderUtils.device->m_LogicalDevice, &renderPassInfo, nullptr,
                               &secondaryRenderPasses->renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create second render pass!");
        }

        //// FrameBuffer setup ////
        if (secondaryRenderPasses->setupFrameBuffer) {
            std::vector<VkImageView> framebufferAttachments{};
            if (secondaryRenderPasses->multisampled) {
                framebufferAttachments.resize(3);
                framebufferAttachments[0] = secondaryRenderPasses->colorImage.view;
                framebufferAttachments[1] = secondaryRenderPasses->depthStencil.view;
                framebufferAttachments[2] = secondaryRenderPasses->colorImage.resolvedView;
            } else {
                framebufferAttachments.resize(2);
                framebufferAttachments[0] = secondaryRenderPasses->colorImage.view;
                framebufferAttachments[1] = secondaryRenderPasses->depthStencil.view;
            }

            VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(m_renderUtils.width,
                                                                                            m_renderUtils.height,
                                                                                            framebufferAttachments.data(),
                                                                                            framebufferAttachments.size(),
                                                                                            secondaryRenderPasses->renderPass);
            secondaryRenderPasses->frameBuffers.resize(m_renderUtils.swapchainImages);
            for (auto &frameBuffer: secondaryRenderPasses->frameBuffers) {
                result = vkCreateFramebuffer(m_renderUtils.device->m_LogicalDevice, &frameBufferCreateInfo, nullptr,
                                             &frameBuffer);
                if (result != VK_SUCCESS) throw std::runtime_error("Failed to create secondary framebuffer");
            }

        }
        VkCommandBuffer copyCmd = m_renderUtils.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);


        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 1;

        if (secondaryRenderPasses->multisampled) {
            Utils::setImageLayout(copyCmd, secondaryRenderPasses->colorImage.resolvedImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else {
            Utils::setImageLayout(copyCmd, secondaryRenderPasses->colorImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }


        VkImageSubresourceRange depthRange = {};
        depthRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthRange.levelCount = 1;
        depthRange.layerCount = 1;

        Utils::setImageLayout(copyCmd, secondaryRenderPasses->depthStencil.image, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, depthRange,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);


        m_renderUtils.device->flushCommandBuffer(copyCmd, m_renderUtils.graphicsQueue, true);


        secondaryRenderPasses->imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        secondaryRenderPasses->imageInfo.imageView = secondaryRenderPasses->colorImage.view; // Your off-screen image view
        secondaryRenderPasses->imageInfo.sampler = secondaryRenderPasses->colorImage.sampler; // The sampler you've just created
        if (secondaryRenderPasses->multisampled) {
            secondaryRenderPasses->imageInfo.imageView = secondaryRenderPasses->colorImage.resolvedView; // Your off-screen image view

        }
        secondaryRenderPasses->depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        secondaryRenderPasses->depthImageInfo.imageView = secondaryRenderPasses->depthStencil.view; // Your off-screen image view
        secondaryRenderPasses->depthImageInfo.sampler = secondaryRenderPasses->depthStencil.sampler; // The sampler you've just created

    }

    void Editor::setupUIRenderPass(EditorRenderPass *secondaryRenderPasses) {
        VkSampleCountFlagBits sampleCount = secondaryRenderPasses->multisampled ? m_renderUtils.msaaSamples
                                                                                : VK_SAMPLE_COUNT_1_BIT;

        //// COLOR IMAGE RESOURCE /////
        VkImageCreateInfo colorImageCI = Populate::imageCreateInfo();
        colorImageCI.imageType = VK_IMAGE_TYPE_2D;
        colorImageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorImageCI.extent = {m_renderUtils.width, m_renderUtils.height, 1};
        colorImageCI.mipLevels = 1;
        colorImageCI.arrayLayers = 1;
        colorImageCI.samples = sampleCount;
        colorImageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorImageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        colorImageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult result = vkCreateImage(m_renderUtils.device->m_LogicalDevice, &colorImageCI, nullptr,
                                        &secondaryRenderPasses->colorImage.image);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create colorImage");

        if (secondaryRenderPasses->multisampled) {
            // Create an additional resolved image if MSAA is used
            VkImageCreateInfo resolvedImageCI = colorImageCI; // Copy from colorImageCI for basic settings
            resolvedImageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            resolvedImageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            result = vkCreateImage(m_renderUtils.device->m_LogicalDevice, &resolvedImageCI, nullptr,
                                   &secondaryRenderPasses->colorImage.resolvedImage);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create resolvedImage");
        }


        {
            VkMemoryRequirements colorMemReqs{};
            vkGetImageMemoryRequirements(m_renderUtils.device->m_LogicalDevice, secondaryRenderPasses->colorImage.image,
                                         &colorMemReqs);

            VkMemoryAllocateInfo colorMemAllloc = Populate::memoryAllocateInfo();
            colorMemAllloc.allocationSize = colorMemReqs.size;
            colorMemAllloc.memoryTypeIndex = m_renderUtils.device->getMemoryType(colorMemReqs.memoryTypeBits,
                                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            result = vkAllocateMemory(m_renderUtils.device->m_LogicalDevice, &colorMemAllloc, nullptr,
                                      &secondaryRenderPasses->colorImage.mem);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
            result = vkBindImageMemory(m_renderUtils.device->m_LogicalDevice, secondaryRenderPasses->colorImage.image,
                                       secondaryRenderPasses->colorImage.mem, 0);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");

        }
        if (secondaryRenderPasses->multisampled) {
            VkMemoryRequirements colorMemReqs{};
            vkGetImageMemoryRequirements(m_renderUtils.device->m_LogicalDevice,
                                         secondaryRenderPasses->colorImage.resolvedImage, &colorMemReqs);

            VkMemoryAllocateInfo colorMemAllloc = Populate::memoryAllocateInfo();
            colorMemAllloc.allocationSize = colorMemReqs.size;
            colorMemAllloc.memoryTypeIndex = m_renderUtils.device->getMemoryType(colorMemReqs.memoryTypeBits,
                                                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            result = vkAllocateMemory(m_renderUtils.device->m_LogicalDevice, &colorMemAllloc, nullptr,
                                      &secondaryRenderPasses->colorImage.resolvedMem);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to allocate depth m_Image memory");
            result = vkBindImageMemory(m_renderUtils.device->m_LogicalDevice,
                                       secondaryRenderPasses->colorImage.resolvedImage,
                                       secondaryRenderPasses->colorImage.resolvedMem, 0);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to bind depth m_Image memory");
        }


        VkImageViewCreateInfo colorImageViewCI = Populate::imageViewCreateInfo();
        colorImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorImageViewCI.image = secondaryRenderPasses->colorImage.image;
        colorImageViewCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        colorImageViewCI.subresourceRange.baseMipLevel = 0;
        colorImageViewCI.subresourceRange.levelCount = 1;
        colorImageViewCI.subresourceRange.baseArrayLayer = 0;
        colorImageViewCI.subresourceRange.layerCount = 1;
        colorImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        // Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT
        if (m_renderUtils.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            colorImageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }

        result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &colorImageViewCI, nullptr,
                                   &secondaryRenderPasses->colorImage.view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create colorImage view");

        if (secondaryRenderPasses->multisampled) {
            colorImageViewCI.image = secondaryRenderPasses->colorImage.resolvedImage;
            result = vkCreateImageView(m_renderUtils.device->m_LogicalDevice, &colorImageViewCI, nullptr,
                                       &secondaryRenderPasses->colorImage.resolvedView);
            if (result != VK_SUCCESS) throw std::runtime_error("Failed to create resolvedVie");
        }
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

        if (vkCreateSampler(m_renderUtils.device->m_LogicalDevice, &samplerInfo, nullptr,
                            &secondaryRenderPasses->colorImage.sampler) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

        VkAttachmentDescription uiColorAttachment = {};
        uiColorAttachment.format = m_renderUtils.swapchainColorFormat;
        uiColorAttachment.samples = m_renderUtils.msaaSamples;
        uiColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Load since it follows the main pass
        uiColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        uiColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiColorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        uiColorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription uiResolveAttachment = {};
        uiResolveAttachment.format = m_renderUtils.swapchainColorFormat;
        uiResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        uiResolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiResolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        uiResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiResolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        uiResolveAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription dummyDepthAttachment = {};
        dummyDepthAttachment.format = m_renderUtils.depthFormat;
        dummyDepthAttachment.samples = m_renderUtils.msaaSamples;
        dummyDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        dummyDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        dummyDepthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        dummyDepthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        dummyDepthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dummyDepthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference uiColorAttachmentRef = {};
        uiColorAttachmentRef.attachment = 0;
        uiColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference uiResolveAttachmentRef = {};
        uiResolveAttachmentRef.attachment = 2;
        uiResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

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

        if (vkCreateRenderPass(m_renderUtils.device->m_LogicalDevice, &uiRenderPassInfo, nullptr,
                               &uiRenderPass.renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create UI render pass!");
        }

        VkCommandBuffer copyCmd = m_renderUtils.device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 1;

        if (secondaryRenderPasses->multisampled) {
            Utils::setImageLayout(copyCmd, secondaryRenderPasses->colorImage.resolvedImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        } else {
            Utils::setImageLayout(copyCmd, secondaryRenderPasses->colorImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }

        m_renderUtils.device->flushCommandBuffer(copyCmd, m_renderUtils.graphicsQueue, true);

        secondaryRenderPasses->imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        secondaryRenderPasses->imageInfo.imageView = secondaryRenderPasses->colorImage.view; // Your off-screen image view
        secondaryRenderPasses->imageInfo.sampler = secondaryRenderPasses->colorImage.sampler; // The sampler you've just created
        if (secondaryRenderPasses->multisampled) {
            secondaryRenderPasses->imageInfo.imageView = secondaryRenderPasses->colorImage.resolvedView; // Your off-screen image view

        }
    }

    void Editor::render() {


    }

}