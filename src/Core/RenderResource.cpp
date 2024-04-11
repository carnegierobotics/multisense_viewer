//
// Created by mgjer on 12/01/2024.
//


#include "Viewer/Core/RenderResource.h"


template<typename T>
RenderResource::GLTFModel<T>::GLTFModel(VkRender::RenderUtils *_renderUtils, std::shared_ptr<T> modelPtr) {
    renderUtils = _renderUtils;
    vulkanDevice = renderUtils->device;
    model = modelPtr;

    generateBRDFLUT();
    setupDescriptors();
    setupRenderPipelines();
}

template<typename T>
void RenderResource::GLTFModel<T>::generateBRDFLUT() {
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
    memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_RESULT(vkAllocateMemory(vulkanDevice->m_LogicalDevice, &memAllocInfo, nullptr, &textures.lutBrdf.m_DeviceMemory));
    CHECK_RESULT(vkBindImageMemory(vulkanDevice->m_LogicalDevice, textures.lutBrdf.m_Image, textures.lutBrdf.m_DeviceMemory, 0));

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
    VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

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
    CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr, &descriptorsetlayout));

    // Pipeline layout
    VkPipelineLayout pipelinelayout;
    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts = &descriptorsetlayout;
    CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr, &pipelinelayout));

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
    blendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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

    std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicStateCI{};
    dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
    dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

    VkPipelineVertexInputStateCreateInfo emptyInputStateCI{};
    emptyInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

    VkShaderModule shaderModuleVert, shaderModuleFrag;
    // Look-up-table (from BRDF) pipeline
    shaderStages = {
            loadShader(vulkanDevice->m_LogicalDevice, "spv/genbrdflut.vert.spv", VK_SHADER_STAGE_VERTEX_BIT, &shaderModuleVert),
            loadShader(vulkanDevice->m_LogicalDevice, "spv/genbrdflut.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT, &shaderModuleFrag)
    };

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



    VkPipeline pipeline;
    CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));

    for (auto shaderStage : shaderStages) {
        vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
    }

    // Render
    VkClearValue clearValues[1];
    clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

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
    viewport.width = (float)dim;
    viewport.height = (float)dim;
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
    Log::Logger::getInstance()->info("Generating BRDF LUT took {} ms", tDiff);

}

template<typename T>
void RenderResource::GLTFModel<T>::setupDescriptors() {
    /*
        Descriptor Pool
    */
    uint32_t imageSamplerCount = 0;
    uint32_t materialCount = 0;
    uint32_t meshCount = 0;

    // Environment samplers (radiance, irradiance, brdf lut)
    imageSamplerCount += 3;

    std::vector<Model*> modellist = { &models.skybox, &models.scene };
    for (auto &model : modellist) {
        for (auto &material : model->materials) {
            imageSamplerCount += 5;
            materialCount++;
        }
        for (auto node : model->linearNodes) {
            if (node->mesh) {
                meshCount++;
            }
        }
    }

    std::vector<VkDescriptorPoolSize> poolSizes = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (4 + meshCount) * swapChain.imageCount },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageSamplerCount * swapChain.imageCount },
            // One SSBO for the shader material buffer
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
    };
    VkDescriptorPoolCreateInfo descriptorPoolCI{};
    descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descriptorPoolCI.pPoolSizes = poolSizes.data();
    descriptorPoolCI.maxSets = (2 + materialCount + meshCount) * swapChain.imageCount;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCI, nullptr, &descriptorPool));

    /*
        Descriptor sets
    */

    // Scene (matrices and environment maps)
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        };
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
        descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
        descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.scene));

        for (auto i = 0; i < descriptorSets.size(); i++) {

            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayouts.scene;
            descriptorSetAllocInfo.descriptorSetCount = 1;
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets[i].scene));

            std::array<VkWriteDescriptorSet, 5> writeDescriptorSets{};

            writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[0].descriptorCount = 1;
            writeDescriptorSets[0].dstSet = descriptorSets[i].scene;
            writeDescriptorSets[0].dstBinding = 0;
            writeDescriptorSets[0].pBufferInfo = &uniformBuffers[i].scene.descriptor;

            writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[1].descriptorCount = 1;
            writeDescriptorSets[1].dstSet = descriptorSets[i].scene;
            writeDescriptorSets[1].dstBinding = 1;
            writeDescriptorSets[1].pBufferInfo = &uniformBuffers[i].params.descriptor;

            writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[2].descriptorCount = 1;
            writeDescriptorSets[2].dstSet = descriptorSets[i].scene;
            writeDescriptorSets[2].dstBinding = 2;
            writeDescriptorSets[2].pImageInfo = &textures.irradianceCube.descriptor;

            writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[3].descriptorCount = 1;
            writeDescriptorSets[3].dstSet = descriptorSets[i].scene;
            writeDescriptorSets[3].dstBinding = 3;
            writeDescriptorSets[3].pImageInfo = &textures.prefilteredCube.descriptor;

            writeDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[4].descriptorCount = 1;
            writeDescriptorSets[4].dstSet = descriptorSets[i].scene;
            writeDescriptorSets[4].dstBinding = 4;
            writeDescriptorSets[4].pImageInfo = &textures.lutBrdf.descriptor;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
        }
    }

    // Material (samplers)
    {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
                { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        };
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
        descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
        descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.material));

        // Per-Material descriptor sets
        for (auto &material : models.scene.materials) {
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayouts.material;
            descriptorSetAllocInfo.descriptorSetCount = 1;
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &material.descriptorSet));

            std::vector<VkDescriptorImageInfo> imageDescriptors = {
                    textures.empty.descriptor,
                    textures.empty.descriptor,
                    material.normalTexture ? material.normalTexture->descriptor : textures.empty.descriptor,
                    material.occlusionTexture ? material.occlusionTexture->descriptor : textures.empty.descriptor,
                    material.emissiveTexture ? material.emissiveTexture->descriptor : textures.empty.descriptor
            };

            // TODO: glTF specs states that metallic roughness should be preferred, even if specular glosiness is present

            if (material.pbrWorkflows.metallicRoughness) {
                if (material.baseColorTexture) {
                    imageDescriptors[0] = material.baseColorTexture->descriptor;
                }
                if (material.metallicRoughnessTexture) {
                    imageDescriptors[1] = material.metallicRoughnessTexture->descriptor;
                }
            }

            if (material.pbrWorkflows.specularGlossiness) {
                if (material.extension.diffuseTexture) {
                    imageDescriptors[0] = material.extension.diffuseTexture->descriptor;
                }
                if (material.extension.specularGlossinessTexture) {
                    imageDescriptors[1] = material.extension.specularGlossinessTexture->descriptor;
                }
            }

            std::array<VkWriteDescriptorSet, 5> writeDescriptorSets{};
            for (size_t i = 0; i < imageDescriptors.size(); i++) {
                writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[i].descriptorCount = 1;
                writeDescriptorSets[i].dstSet = material.descriptorSet;
                writeDescriptorSets[i].dstBinding = static_cast<uint32_t>(i);
                writeDescriptorSets[i].pImageInfo = &imageDescriptors[i];
            }

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
        }

        // Model node (matrices)
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.node));

            // Per-Node descriptor set
            for (auto &node : models.scene.nodes) {
                setupNodeDescriptorSet(node);
            }
        }

        // Material Buffer
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                    { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.materialBuffer));

            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayouts.materialBuffer;
            descriptorSetAllocInfo.descriptorSetCount = 1;
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSetMaterials));

            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = descriptorSetMaterials;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &shaderMaterialBuffer.descriptor;
            vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
        }

    }

    // Skybox (fixed set)
    for (auto i = 0; i < uniformBuffers.size(); i++) {
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayouts.scene;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets[i].skybox));

        std::array<VkWriteDescriptorSet, 3> writeDescriptorSets{};

        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = descriptorSets[i].skybox;
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &uniformBuffers[i].skybox.descriptor;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = descriptorSets[i].skybox;
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &uniformBuffers[i].params.descriptor;

        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].dstSet = descriptorSets[i].skybox;
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].pImageInfo = &textures.prefilteredCube.descriptor;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
    }
}

template<typename T>
void RenderResource::GLTFModel<T>::setupRenderPipelines() {

}
