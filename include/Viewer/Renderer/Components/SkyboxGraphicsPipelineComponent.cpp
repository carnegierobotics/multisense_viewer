//
// Created by mgjer on 12/01/2024.
//


#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/Tools/Macros.h"

namespace RenderResource {

    void SkyboxGraphicsPipelineComponent::generateBRDFLUT() {
        auto tStart = std::chrono::high_resolution_clock::now();

        const VkFormat format = VK_FORMAT_R16G16_SFLOAT;
        const int32_t dim = 512;

        // Image
        VkImageCreateInfo imageCI{};
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        CHECK_RESULT(vkCreateImage(vulkanDevice->m_LogicalDevice, &imageCI, nullptr, &textures.lutBrdf.m_Image));
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(vulkanDevice->m_LogicalDevice, textures.lutBrdf.m_Image, &memReqs);
        VkMemoryAllocateInfo memAllocInfo{};
        memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memAllocInfo.allocationSize = memReqs.size;
        memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        CHECK_RESULT(
                vkAllocateMemory(vulkanDevice->m_LogicalDevice, &memAllocInfo, nullptr,
                                 &textures.lutBrdf.m_DeviceMemory));
        CHECK_RESULT(
                vkBindImageMemory(vulkanDevice->m_LogicalDevice, textures.lutBrdf.m_Image,
                                  textures.lutBrdf.m_DeviceMemory,
                                  0));

        // View
        VkImageViewCreateInfo viewCI{};
        viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format = format;
        viewCI.subresourceRange = {};
        viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.levelCount = 1;
        viewCI.subresourceRange.layerCount = 1;
        viewCI.image = textures.lutBrdf.m_Image;
        CHECK_RESULT(vkCreateImageView(vulkanDevice->m_LogicalDevice, &viewCI, nullptr, &textures.lutBrdf.m_View));

        // Sampler
        VkSamplerCreateInfo samplerCI{};
        samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCI.magFilter = VK_FILTER_LINEAR;
        samplerCI.minFilter = VK_FILTER_LINEAR;
        samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCI.minLod = 0.0f;
        samplerCI.maxLod = 1.0f;
        samplerCI.maxAnisotropy = 1.0f;
        samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        CHECK_RESULT(vkCreateSampler(vulkanDevice->m_LogicalDevice, &samplerCI, nullptr, &textures.lutBrdf.m_Sampler));

        // FB, Att, RP, Pipe, etc.
        VkAttachmentDescription attDesc{};
        // Color attachment
        attDesc.format = format;
        attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpassDescription{};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Use subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;
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

        // Create the actual renderpass
        VkRenderPassCreateInfo renderPassCI{};
        renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassCI.attachmentCount = 1;
        renderPassCI.pAttachments = &attDesc;
        renderPassCI.subpassCount = 1;
        renderPassCI.pSubpasses = &subpassDescription;
        renderPassCI.dependencyCount = 2;
        renderPassCI.pDependencies = dependencies.data();

        VkRenderPass renderpass;
        CHECK_RESULT(vkCreateRenderPass(vulkanDevice->m_LogicalDevice, &renderPassCI, nullptr, &renderpass));

        VkFramebufferCreateInfo framebufferCI{};
        framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferCI.renderPass = renderpass;
        framebufferCI.attachmentCount = 1;
        framebufferCI.pAttachments = &textures.lutBrdf.m_View;
        framebufferCI.width = dim;
        framebufferCI.height = dim;
        framebufferCI.layers = 1;

        VkFramebuffer framebuffer;
        CHECK_RESULT(vkCreateFramebuffer(vulkanDevice->m_LogicalDevice, &framebufferCI, nullptr, &framebuffer));

        // Desriptors
        VkDescriptorSetLayout descriptorsetlayout;
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
        descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                                 &descriptorsetlayout));

        // Pipeline layout
        VkPipelineLayout pipelinelayout;
        VkPipelineLayoutCreateInfo pipelineLayoutCI{};
        pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCI.setLayoutCount = 1;
        pipelineLayoutCI.pSetLayouts = &descriptorsetlayout;
        CHECK_RESULT(
                vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr, &pipelinelayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
        inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
        rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
        rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationStateCI.lineWidth = 1.0f;

        VkPipelineColorBlendAttachmentState blendAttachmentState{};
        blendAttachmentState.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        blendAttachmentState.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
        colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendStateCI.attachmentCount = 1;
        colorBlendStateCI.pAttachments = &blendAttachmentState;

        VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
        depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilStateCI.depthTestEnable = VK_FALSE;
        depthStencilStateCI.depthWriteEnable = VK_FALSE;
        depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencilStateCI.front = depthStencilStateCI.back;
        depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;

        VkPipelineViewportStateCreateInfo viewportStateCI{};
        viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateCI.viewportCount = 1;
        viewportStateCI.scissorCount = 1;

        VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
        multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicStateCI{};
        dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
        dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

        VkPipelineVertexInputStateCreateInfo emptyInputStateCI{};
        emptyInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.layout = pipelinelayout;
        pipelineCI.renderPass = renderpass;
        pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
        pipelineCI.pVertexInputState = &emptyInputStateCI;
        pipelineCI.pRasterizationState = &rasterizationStateCI;
        pipelineCI.pColorBlendState = &colorBlendStateCI;
        pipelineCI.pMultisampleState = &multisampleStateCI;
        pipelineCI.pViewportState = &viewportStateCI;
        pipelineCI.pDepthStencilState = &depthStencilStateCI;
        pipelineCI.pDynamicState = &dynamicStateCI;
        pipelineCI.stageCount = 2;
        pipelineCI.pStages = shaderStages.data();


        VkShaderModule vertModule{};
        VkShaderModule fragModule{};
        // Look-up-table (from BRDF) pipeline
        shaderStages = {
                loadShader("spv/genbrdflut.vert.spv", VK_SHADER_STAGE_VERTEX_BIT, &vertModule),
                loadShader("spv/genbrdflut.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule)
        };
        VkPipeline pipeline;
        CHECK_RESULT(
                vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));
        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
        }

        // Render
        VkClearValue clearValues[1];
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderPass = renderpass;
        renderPassBeginInfo.renderArea.extent.width = dim;
        renderPassBeginInfo.renderArea.extent.height = dim;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = clearValues;
        renderPassBeginInfo.framebuffer = framebuffer;

        VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.width = (float) dim;
        viewport.height = (float) dim;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.extent.width = dim;
        scissor.extent.height = dim;

        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdDraw(cmdBuf, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmdBuf);
        vulkanDevice->flushCommandBuffer(cmdBuf, vulkanDevice->m_TransferQueue);

        vkQueueWaitIdle(vulkanDevice->m_TransferQueue);

        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelinelayout, nullptr);
        vkDestroyRenderPass(vulkanDevice->m_LogicalDevice, renderpass, nullptr);
        vkDestroyFramebuffer(vulkanDevice->m_LogicalDevice, framebuffer, nullptr);
        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorsetlayout, nullptr);

        textures.lutBrdf.m_Descriptor.imageView = textures.lutBrdf.m_View;
        textures.lutBrdf.m_Descriptor.sampler = textures.lutBrdf.m_Sampler;
        textures.lutBrdf.m_Descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        textures.lutBrdf.m_Device = vulkanDevice;

        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        Log::Logger::getInstance()->info("Generating BRDF LUT took {}ms", tDiff);

    }

    void SkyboxGraphicsPipelineComponent::generateCubemaps(const VkRender::GLTFModelComponent &cube) {
        enum Target {
            IRRADIANCE = 0, PREFILTEREDENV = 1
        };

        for (uint32_t target = 0; target < PREFILTEREDENV + 1; target++) {

            TextureCubeMap cubemap;

            auto tStart = std::chrono::high_resolution_clock::now();

            VkFormat format;
            int32_t dim;

            switch (target) {
                case IRRADIANCE:
                    format = VK_FORMAT_R32G32B32A32_SFLOAT;
                    dim = 64;
                    break;
                case PREFILTEREDENV:
                    format = VK_FORMAT_R16G16B16A16_SFLOAT;
                    dim = 512;
                    break;
            };

            const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

            // Create target cubemap
            {
                // Image
                VkImageCreateInfo imageCI{};
                imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                imageCI.format = format;
                imageCI.extent.width = dim;
                imageCI.extent.height = dim;
                imageCI.extent.depth = 1;
                imageCI.mipLevels = numMips;
                imageCI.arrayLayers = 6;
                imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
                imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
                imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                CHECK_RESULT(vkCreateImage(vulkanDevice->m_LogicalDevice, &imageCI, nullptr, &cubemap.m_Image));
                VkMemoryRequirements memReqs;
                vkGetImageMemoryRequirements(vulkanDevice->m_LogicalDevice, cubemap.m_Image, &memReqs);
                VkMemoryAllocateInfo memAllocInfo{};
                memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                memAllocInfo.allocationSize = memReqs.size;
                memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                CHECK_RESULT(
                        vkAllocateMemory(vulkanDevice->m_LogicalDevice, &memAllocInfo, nullptr,
                                         &cubemap.m_DeviceMemory));
                CHECK_RESULT(
                        vkBindImageMemory(vulkanDevice->m_LogicalDevice, cubemap.m_Image, cubemap.m_DeviceMemory, 0));

                // View
                VkImageViewCreateInfo viewCI{};
                viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
                viewCI.format = format;
                viewCI.subresourceRange = {};
                viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                viewCI.subresourceRange.levelCount = numMips;
                viewCI.subresourceRange.layerCount = 6;
                viewCI.image = cubemap.m_Image;
                CHECK_RESULT(vkCreateImageView(vulkanDevice->m_LogicalDevice, &viewCI, nullptr, &cubemap.m_View));

                // Sampler
                VkSamplerCreateInfo samplerCI{};
                samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                samplerCI.magFilter = VK_FILTER_LINEAR;
                samplerCI.minFilter = VK_FILTER_LINEAR;
                samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                samplerCI.minLod = 0.0f;
                samplerCI.maxLod = static_cast<float>(numMips);
                samplerCI.maxAnisotropy = 1.0f;
                samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
                CHECK_RESULT(vkCreateSampler(vulkanDevice->m_LogicalDevice, &samplerCI, nullptr, &cubemap.m_Sampler));
            }

            // FB, Att, RP, Pipe, etc.
            VkAttachmentDescription attDesc{};
            // Color attachment
            attDesc.format = format;
            attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
            attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

            VkSubpassDescription subpassDescription{};
            subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpassDescription.colorAttachmentCount = 1;
            subpassDescription.pColorAttachments = &colorReference;

            // Use subpass dependencies for layout transitions
            std::array<VkSubpassDependency, 2> dependencies;
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

            // Renderpass
            VkRenderPassCreateInfo renderPassCI{};
            renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassCI.attachmentCount = 1;
            renderPassCI.pAttachments = &attDesc;
            renderPassCI.subpassCount = 1;
            renderPassCI.pSubpasses = &subpassDescription;
            renderPassCI.dependencyCount = 2;
            renderPassCI.pDependencies = dependencies.data();
            VkRenderPass renderpass;
            CHECK_RESULT(vkCreateRenderPass(vulkanDevice->m_LogicalDevice, &renderPassCI, nullptr, &renderpass));

            struct Offscreen {
                VkImage image;
                VkImageView view;
                VkDeviceMemory memory;
                VkFramebuffer framebuffer;
            } offscreen;

            // Create offscreen framebuffer
            {
                // Image
                VkImageCreateInfo imageCI{};
                imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                imageCI.imageType = VK_IMAGE_TYPE_2D;
                imageCI.format = format;
                imageCI.extent.width = dim;
                imageCI.extent.height = dim;
                imageCI.extent.depth = 1;
                imageCI.mipLevels = 1;
                imageCI.arrayLayers = 1;
                imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
                imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
                imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                CHECK_RESULT(vkCreateImage(vulkanDevice->m_LogicalDevice, &imageCI, nullptr, &offscreen.image));
                VkMemoryRequirements memReqs;
                vkGetImageMemoryRequirements(vulkanDevice->m_LogicalDevice, offscreen.image, &memReqs);
                VkMemoryAllocateInfo memAllocInfo{};
                memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                memAllocInfo.allocationSize = memReqs.size;
                memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                CHECK_RESULT(
                        vkAllocateMemory(vulkanDevice->m_LogicalDevice, &memAllocInfo, nullptr, &offscreen.memory));
                CHECK_RESULT(vkBindImageMemory(vulkanDevice->m_LogicalDevice, offscreen.image, offscreen.memory, 0));

                // View
                VkImageViewCreateInfo viewCI{};
                viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewCI.format = format;
                viewCI.flags = 0;
                viewCI.subresourceRange = {};
                viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                viewCI.subresourceRange.baseMipLevel = 0;
                viewCI.subresourceRange.levelCount = 1;
                viewCI.subresourceRange.baseArrayLayer = 0;
                viewCI.subresourceRange.layerCount = 1;
                viewCI.image = offscreen.image;
                CHECK_RESULT(vkCreateImageView(vulkanDevice->m_LogicalDevice, &viewCI, nullptr, &offscreen.view));

                // Framebuffer
                VkFramebufferCreateInfo framebufferCI{};
                framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferCI.renderPass = renderpass;
                framebufferCI.attachmentCount = 1;
                framebufferCI.pAttachments = &offscreen.view;
                framebufferCI.width = dim;
                framebufferCI.height = dim;
                framebufferCI.layers = 1;
                CHECK_RESULT(vkCreateFramebuffer(vulkanDevice->m_LogicalDevice, &framebufferCI, nullptr,
                                                 &offscreen.framebuffer));

                VkCommandBuffer layoutCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
                VkImageMemoryBarrier imageMemoryBarrier{};
                imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imageMemoryBarrier.image = offscreen.image;
                imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                imageMemoryBarrier.srcAccessMask = 0;
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(layoutCmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     0,
                                     0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
                vulkanDevice->flushCommandBuffer(layoutCmd, vulkanDevice->m_TransferQueue, true);
            }

            // Descriptors
            VkDescriptorSetLayout descriptorsetlayout;
            VkDescriptorSetLayoutBinding setLayoutBinding = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = &setLayoutBinding;
            descriptorSetLayoutCI.bindingCount = 1;
            CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                                     &descriptorsetlayout));

            // Descriptor Pool
            VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
            VkDescriptorPoolCreateInfo descriptorPoolCI{};
            descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCI.poolSizeCount = 1;
            descriptorPoolCI.pPoolSizes = &poolSize;
            descriptorPoolCI.maxSets = 2;
            VkDescriptorPool descriptorpool;
            CHECK_RESULT(
                    vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &descriptorPoolCI, nullptr, &descriptorpool));

            // Descriptor sets
            VkDescriptorSet descriptorset;
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorpool;
            descriptorSetAllocInfo.pSetLayouts = &descriptorsetlayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
            CHECK_RESULT(
                    vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo, &descriptorset));
            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = descriptorset;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pImageInfo = &textures.environmentCube.m_Descriptor;
            vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, 1, &writeDescriptorSet, 0, nullptr);

            struct PushBlockIrradiance {
                glm::mat4 mvp;
                float deltaPhi = (2.0f * float(M_PI)) / 180.0f;
                float deltaTheta = (0.5f * float(M_PI)) / 64.0f;
            } pushBlockIrradiance;

            struct PushBlockPrefilterEnv {
                glm::mat4 mvp;
                float roughness;
                uint32_t numSamples = 32u;
            } pushBlockPrefilterEnv;

            // Pipeline layout
            VkPipelineLayout pipelinelayout;
            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            switch (target) {
                case IRRADIANCE:
                    pushConstantRange.size = sizeof(PushBlockIrradiance);
                    break;
                case PREFILTEREDENV:
                    pushConstantRange.size = sizeof(PushBlockPrefilterEnv);
                    break;
            };

            VkPipelineLayoutCreateInfo pipelineLayoutCI{};
            pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCI.setLayoutCount = 1;
            pipelineLayoutCI.pSetLayouts = &descriptorsetlayout;
            pipelineLayoutCI.pushConstantRangeCount = 1;
            pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
            CHECK_RESULT(
                    vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr, &pipelinelayout));

            // Pipeline
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
            inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
            rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
            rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterizationStateCI.lineWidth = 1.0f;

            VkPipelineColorBlendAttachmentState blendAttachmentState{};
            blendAttachmentState.colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                    VK_COLOR_COMPONENT_A_BIT;
            blendAttachmentState.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
            colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlendStateCI.attachmentCount = 1;
            colorBlendStateCI.pAttachments = &blendAttachmentState;

            VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
            depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencilStateCI.depthTestEnable = VK_FALSE;
            depthStencilStateCI.depthWriteEnable = VK_FALSE;
            depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            depthStencilStateCI.front = depthStencilStateCI.back;
            depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;

            VkPipelineViewportStateCreateInfo viewportStateCI{};
            viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportStateCI.viewportCount = 1;
            viewportStateCI.scissorCount = 1;

            VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
            multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamicStateCI{};
            dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
            dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

            // Vertex input state
            VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(VkRender::Vertex),
                                                                  VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription vertexInputAttribute = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};

            VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
            vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputStateCI.vertexBindingDescriptionCount = 1;
            vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
            vertexInputStateCI.vertexAttributeDescriptionCount = 1;
            vertexInputStateCI.pVertexAttributeDescriptions = &vertexInputAttribute;

            std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

            VkGraphicsPipelineCreateInfo pipelineCI{};
            pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCI.layout = pipelinelayout;
            pipelineCI.renderPass = renderpass;
            pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
            pipelineCI.pVertexInputState = &vertexInputStateCI;
            pipelineCI.pRasterizationState = &rasterizationStateCI;
            pipelineCI.pColorBlendState = &colorBlendStateCI;
            pipelineCI.pMultisampleState = &multisampleStateCI;
            pipelineCI.pViewportState = &viewportStateCI;
            pipelineCI.pDepthStencilState = &depthStencilStateCI;
            pipelineCI.pDynamicState = &dynamicStateCI;
            pipelineCI.stageCount = 2;
            pipelineCI.pStages = shaderStages.data();
            pipelineCI.renderPass = renderpass;

            VkShaderModule vertModule{};
            VkShaderModule fragModule{};

            shaderStages[0] = loadShader("spv/filtercube.vert.spv", VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
            switch (target) {
                case IRRADIANCE:
                    shaderStages[1] = loadShader("spv/irradiancecube.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT,
                                                 &fragModule);
                    break;
                case PREFILTEREDENV:
                    shaderStages[1] = loadShader("spv/prefilterenvmap.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT,
                                                 &fragModule);
                    break;
            };
            VkPipeline pipeline;
            CHECK_RESULT(
                    vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                              &pipeline));
            for (auto shaderStage: shaderStages) {
                vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
            }

            // Render cubemap
            VkClearValue clearValues[1];
            clearValues[0].color = {{0.0f, 0.0f, 0.2f, 0.0f}};

            VkRenderPassBeginInfo renderPassBeginInfo{};
            renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassBeginInfo.renderPass = renderpass;
            renderPassBeginInfo.framebuffer = offscreen.framebuffer;
            renderPassBeginInfo.renderArea.extent.width = dim;
            renderPassBeginInfo.renderArea.extent.height = dim;
            renderPassBeginInfo.clearValueCount = 1;
            renderPassBeginInfo.pClearValues = clearValues;

            std::vector<glm::mat4> matrices = {
                    glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
                                glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                    glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
                                glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                    glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                    glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                    glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
                    glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
            };

            VkCommandBuffer cmdBuf = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, false);

            VkViewport viewport{};
            viewport.width = dim;
            viewport.height = dim;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor{};
            scissor.extent.width = dim;
            scissor.extent.height = dim;

            VkImageSubresourceRange subresourceRange{};
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourceRange.baseMipLevel = 0;
            subresourceRange.levelCount = numMips;
            subresourceRange.layerCount = 6;

            // Change image layout for all cubemap faces to transfer destination
            {
                vulkanDevice->beginCommandBuffer(cmdBuf);
                VkImageMemoryBarrier imageMemoryBarrier{};
                imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imageMemoryBarrier.image = cubemap.m_Image;
                imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                imageMemoryBarrier.srcAccessMask = 0;
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imageMemoryBarrier.subresourceRange = subresourceRange;
                vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                     0,
                                     nullptr, 0, nullptr, 1, &imageMemoryBarrier);
                vulkanDevice->flushCommandBuffer(cmdBuf, vulkanDevice->m_TransferQueue, false);
            }

            for (uint32_t m = 0; m < numMips; m++) {
                for (uint32_t f = 0; f < 6; f++) {

                    vulkanDevice->beginCommandBuffer(cmdBuf);

                    viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                    viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
                    vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

                    // Render scene from cube face's point of view
                    vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

                    // Pass parameters for current pass using a push constant block
                    switch (target) {
                        case IRRADIANCE:
                            pushBlockIrradiance.mvp =
                                    glm::perspective((float) (M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];
                            vkCmdPushConstants(cmdBuf, pipelinelayout,
                                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                               sizeof(PushBlockIrradiance), &pushBlockIrradiance);
                            break;
                        case PREFILTEREDENV:
                            pushBlockPrefilterEnv.mvp =
                                    glm::perspective((float) (M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];
                            pushBlockPrefilterEnv.roughness = (float) m / (float) (numMips - 1);
                            vkCmdPushConstants(cmdBuf, pipelinelayout,
                                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                               sizeof(PushBlockPrefilterEnv), &pushBlockPrefilterEnv);
                            break;
                    };

                    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinelayout, 0, 1,
                                            &descriptorset,
                                            0, NULL);

                    VkDeviceSize offsets[1] = {0};

                    cube.model->draw(cmdBuf); // TODO

                    vkCmdEndRenderPass(cmdBuf);

                    VkImageSubresourceRange subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    subresourceRange.baseMipLevel = 0;
                    subresourceRange.levelCount = numMips;
                    subresourceRange.layerCount = 6;

                    {
                        VkImageMemoryBarrier imageMemoryBarrier{};
                        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                        imageMemoryBarrier.image = offscreen.image;
                        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                        imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
                    }

                    // Copy region for transfer from framebuffer to cube face
                    VkImageCopy copyRegion{};

                    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    copyRegion.srcSubresource.baseArrayLayer = 0;
                    copyRegion.srcSubresource.mipLevel = 0;
                    copyRegion.srcSubresource.layerCount = 1;
                    copyRegion.srcOffset = {0, 0, 0};

                    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    copyRegion.dstSubresource.baseArrayLayer = f;
                    copyRegion.dstSubresource.mipLevel = m;
                    copyRegion.dstSubresource.layerCount = 1;
                    copyRegion.dstOffset = {0, 0, 0};

                    copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
                    copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
                    copyRegion.extent.depth = 1;

                    vkCmdCopyImage(
                            cmdBuf,
                            offscreen.image,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            cubemap.m_Image,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1,
                            &copyRegion);

                    {
                        VkImageMemoryBarrier imageMemoryBarrier{};
                        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                        imageMemoryBarrier.image = offscreen.image;
                        imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                        imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
                    }

                    vulkanDevice->flushCommandBuffer(cmdBuf, vulkanDevice->m_TransferQueue, false);
                }
            }

            {
                vulkanDevice->beginCommandBuffer(cmdBuf);
                VkImageMemoryBarrier imageMemoryBarrier{};
                imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imageMemoryBarrier.image = cubemap.m_Image;
                imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                imageMemoryBarrier.subresourceRange = subresourceRange;
                vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                     0,
                                     nullptr, 0, nullptr, 1, &imageMemoryBarrier);
                vulkanDevice->flushCommandBuffer(cmdBuf, vulkanDevice->m_TransferQueue, false);
            }


            vkDestroyRenderPass(vulkanDevice->m_LogicalDevice, renderpass, nullptr);
            vkDestroyFramebuffer(vulkanDevice->m_LogicalDevice, offscreen.framebuffer, nullptr);
            vkFreeMemory(vulkanDevice->m_LogicalDevice, offscreen.memory, nullptr);
            vkDestroyImageView(vulkanDevice->m_LogicalDevice, offscreen.view, nullptr);
            vkDestroyImage(vulkanDevice->m_LogicalDevice, offscreen.image, nullptr);
            vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorpool, nullptr);
            vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorsetlayout, nullptr);
            vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);
            vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelinelayout, nullptr);

            cubemap.m_Descriptor.imageView = cubemap.m_View;
            cubemap.m_Descriptor.sampler = cubemap.m_Sampler;
            cubemap.m_Descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            cubemap.m_Device = vulkanDevice;

            switch (target) {
                case IRRADIANCE:
                    textures.irradianceCube = cubemap;
                    break;
                case PREFILTEREDENV:
                    textures.prefilteredCube = cubemap;
                    shaderValuesParams.prefilteredCubeMipLevels = static_cast<float>(numMips);
                    break;
            };

            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            Log::Logger::getInstance()->info("Generated cube map from enviroment file. Took {} for {} mips", tDiff,
                                             numMips);
        }
    }

    void SkyboxGraphicsPipelineComponent::setupUniformBuffers() {
        bufferSkyboxVert.resize(renderUtils->UBCount);
        bufferSkyboxFrag.resize(renderUtils->UBCount);
        for (size_t i = 0; i < renderUtils->UBCount; ++i) {
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                       &bufferSkyboxFrag[i], sizeof(ShaderValuesParams));
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                       &bufferSkyboxVert[i], sizeof(VkRender::UBOMatrix));

            bufferSkyboxVert[i].map();
            bufferSkyboxFrag[i].map();

        }
    }

    void
    SkyboxGraphicsPipelineComponent::setupDescriptors(
            const VkRender::GLTFModelComponent &gltfComponent) {
/*
			Descriptor Pool
		*/
        uint32_t imageSamplerCount = 0;
        uint32_t materialCount = 0;
        uint32_t meshCount = 0;

        // Environment samplers (radiance, irradiance, brdf lut)
        imageSamplerCount += 3;

        for (auto &material: gltfComponent.model->materials) {
            imageSamplerCount += 5;
            materialCount++;
        }

        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         (4 + meshCount) * renderUtils->UBCount},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageSamplerCount * renderUtils->UBCount},
                // One SSBO for the shader material buffer
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1}
        };
        VkDescriptorPoolCreateInfo descriptorPoolCI{};
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        descriptorPoolCI.pPoolSizes = poolSizes.data();
        descriptorPoolCI.maxSets = (2 + materialCount + meshCount) * renderUtils->UBCount;
        CHECK_RESULT(
                vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &descriptorPoolCI, nullptr, &descriptorPool));

        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1,
                                                                  VK_SHADER_STAGE_VERTEX_BIT |
                                                                  VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
        descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
        descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
        CHECK_RESULT(
                vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                            &setLayout));

        descriptorSets.resize(renderUtils->UBCount);
        // Skybox (fixed set)
        for (auto i = 0; i < renderUtils->UBCount; i++) {
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &setLayout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
            CHECK_RESULT(
                    vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                             &descriptorSets[i]));

            std::array<VkWriteDescriptorSet, 3> writeDescriptorSets{};

            writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[0].descriptorCount = 1;
            writeDescriptorSets[0].dstSet = descriptorSets[i];
            writeDescriptorSets[0].dstBinding = 0;
            writeDescriptorSets[0].pBufferInfo = &bufferSkyboxVert[i].m_DescriptorBufferInfo;

            writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[1].descriptorCount = 1;
            writeDescriptorSets[1].dstSet = descriptorSets[i];
            writeDescriptorSets[1].dstBinding = 1;
            writeDescriptorSets[1].pBufferInfo = &bufferSkyboxFrag[i].m_DescriptorBufferInfo;

            writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[2].descriptorCount = 1;
            writeDescriptorSets[2].dstSet = descriptorSets[i];
            writeDescriptorSets[2].dstBinding = 2;
            writeDescriptorSets[2].pImageInfo = &textures.prefilteredCube.m_Descriptor;

            vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                   writeDescriptorSets.data(), 0, nullptr);
        }
    }

    void SkyboxGraphicsPipelineComponent::setupPipelines() {
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
        inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
        rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
        rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationStateCI.lineWidth = 1.0f;

        VkPipelineColorBlendAttachmentState blendAttachmentState{};
        blendAttachmentState.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        blendAttachmentState.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
        colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendStateCI.attachmentCount = 1;
        colorBlendStateCI.pAttachments = &blendAttachmentState;

        VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
        depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilStateCI.depthTestEnable = VK_FALSE;
        depthStencilStateCI.depthWriteEnable = VK_FALSE;
        depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencilStateCI.front = depthStencilStateCI.back;
        depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;

        VkPipelineViewportStateCreateInfo viewportStateCI{};
        viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateCI.viewportCount = 1;
        viewportStateCI.scissorCount = 1;

        VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
        multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleStateCI.rasterizationSamples = renderUtils->msaaSamples;


        std::vector<VkDynamicState> dynamicStateEnables = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicStateCI{};
        dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
        dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

        // Pipeline layout
        const std::vector<VkDescriptorSetLayout> setLayouts = {
                setLayout
        };
        VkPipelineLayoutCreateInfo pipelineLayoutCI{};
        pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCI.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
        pipelineLayoutCI.pSetLayouts = setLayouts.data();
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.size = sizeof(uint32_t);
        pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pipelineLayoutCI.pushConstantRangeCount = 1;
        pipelineLayoutCI.pPushConstantRanges = &pushConstantRange;
        CHECK_RESULT(
                vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr, &pipelineLayout));

        // Vertex bindings an attributes
        VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(VkRender::Vertex),
                                                              VK_VERTEX_INPUT_RATE_VERTEX};
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6},
                {3, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 8},
        };
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCI.vertexBindingDescriptionCount = 1;
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

        // Pipelines
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.layout = pipelineLayout;
        pipelineCI.renderPass = *renderUtils->renderPass;
        pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
        pipelineCI.pVertexInputState = &vertexInputStateCI;
        pipelineCI.pRasterizationState = &rasterizationStateCI;
        pipelineCI.pColorBlendState = &colorBlendStateCI;
        pipelineCI.pMultisampleState = &multisampleStateCI;
        pipelineCI.pViewportState = &viewportStateCI;
        pipelineCI.pDepthStencilState = &depthStencilStateCI;
        pipelineCI.pDynamicState = &dynamicStateCI;
        pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCI.pStages = shaderStages.data();

        VkShaderModule vertModule, fragModule;

        shaderStages[0] = loadShader("spv/skybox.vert.spv", VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = loadShader("spv/skybox.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        // Default pipeline with back-face culling
        CHECK_RESULT(
                vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));


        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
        }

    }

    void SkyboxGraphicsPipelineComponent::update() {
        memcpy(bufferSkyboxFrag[renderUtils->swapchainIndex].mapped, &shaderValuesParams, sizeof(ShaderValuesParams));
        memcpy(bufferSkyboxVert[renderUtils->swapchainIndex].mapped, &uboMatrix, sizeof(VkRender::UBOMatrix));

    }

    void SkyboxGraphicsPipelineComponent::draw(CommandBuffer *commandBuffer, uint32_t cbIndex) {

        vkCmdBindDescriptorSets(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &descriptorSets[cbIndex], 0, nullptr);
        vkCmdBindPipeline(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

};
