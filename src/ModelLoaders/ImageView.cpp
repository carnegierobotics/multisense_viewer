//
// Created by magnus on 4/8/24.
//

#include "Viewer/ModelLoaders/ImageView.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Core/CommandBuffer.h"


ImageView::Model::Model(uint32_t framesInFlight, VulkanDevice *vulkanDevice) {
    m_vulkanDevice = vulkanDevice;
    m_framesInFlight = framesInFlight;
    m_mesh.vertices.resize(framesInFlight);
    m_mesh.indices.resize(framesInFlight);

    int numRenderPasses = 2; // Todo make adjustable
    resources.resize(numRenderPasses);
    for (int i = 0; i < numRenderPasses; ++i) {
        resources[i].texture.resize(framesInFlight);
        resources[i].descriptors.resize(framesInFlight);
        resources[i].descriptorSetLayout.resize(framesInFlight);
        resources[i].pipelineLayout.resize(framesInFlight);
        resources[i].pipeline.resize(framesInFlight);
    }


}

ImageView::Model::~Model() {

    if (m_vulkanDevice != nullptr) {
        for (uint32_t j = 0; j < m_framesInFlight; ++j) {
            vkFreeMemory(m_vulkanDevice->m_LogicalDevice, m_mesh.vertices[j].memory, nullptr);
            vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, m_mesh.vertices[j].buffer, nullptr);
            if (m_mesh.indexCount > 0) {
                vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, m_mesh.indices[j].buffer, nullptr);
                vkFreeMemory(m_vulkanDevice->m_LogicalDevice, m_mesh.indices[j].memory, nullptr);
            }
        }

        int numRenderPasses = 2; // Todo make adjustable
        for (int i = 0; i < numRenderPasses; ++i) {
            for (uint32_t j = 0; j < m_framesInFlight; ++j) {

                vkDestroyPipeline(m_vulkanDevice->m_LogicalDevice, resources[i].pipeline[j], nullptr);
                vkDestroyDescriptorSetLayout(m_vulkanDevice->m_LogicalDevice, resources[i].descriptorSetLayout[j],
                                             nullptr);
                vkDestroyPipelineLayout(m_vulkanDevice->m_LogicalDevice, resources[i].pipelineLayout[j], nullptr);
            }
            vkDestroyDescriptorPool(m_vulkanDevice->m_LogicalDevice, resources[i].descriptorPool, nullptr);
        }
    }
}

void
ImageView::Model::createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                        const std::vector<uint32_t> &indices) {
    size_t vertexBufferSize = vertices.size() * sizeof(VkRender::Vertex);
    size_t indexBufferSize = indices.size() * sizeof(uint32_t);
    m_mesh.vertexCount = static_cast<uint32_t>(vertices.size());
    m_mesh.indexCount = static_cast<uint32_t>(indices.size());

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging{}, indexStaging{};

    // Create staging buffers
    // Vertex m_DataPtr
    CHECK_RESULT(m_vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            reinterpret_cast<const void *>(vertices.data())))
    // Index m_DataPtr
    if (indexBufferSize > 0) {
        CHECK_RESULT(m_vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                reinterpret_cast<const void *>(indices.data())))
    }
    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        // Create m_Device local buffers
        // Vertex buffer
        if (m_mesh.vertices[i].buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, m_mesh.vertices[i].buffer, nullptr);
            vkFreeMemory(m_vulkanDevice->m_LogicalDevice, m_mesh.vertices[i].memory, nullptr);
            if (indexBufferSize > 0) {
                vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, m_mesh.indices[i].buffer, nullptr);
                vkFreeMemory(m_vulkanDevice->m_LogicalDevice, m_mesh.indices[i].memory, nullptr);
            }
        }
        CHECK_RESULT(m_vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &m_mesh.vertices[i].buffer,
                &m_mesh.vertices[i].memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &m_mesh.indices[i].buffer,
                    &m_mesh.indices[i].memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, m_mesh.vertices[i].buffer, 1, &copyRegion);
        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, m_mesh.indices[i].buffer, 1, &copyRegion);
        }
        m_vulkanDevice->flushCommandBuffer(copyCmd, m_vulkanDevice->m_TransferQueue, true);
    }
    vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(m_vulkanDevice->m_LogicalDevice, vertexStaging.memory, nullptr);

    if (indexBufferSize > 0) {
        vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, indexStaging.buffer, nullptr);
        vkFreeMemory(m_vulkanDevice->m_LogicalDevice, indexStaging.memory, nullptr);
    }
}


void ImageView::Model::createEmptyTexture(uint32_t width, uint32_t height) {
    Log::Logger::getInstance()->info("Preparing ImageView Texture image {}, {}, with", width, height);
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    int numRenderPasses = 2; // Todo make adjustable
    for (int j = 0; j < numRenderPasses; ++j) {
        for (uint32_t i = 0; i < m_framesInFlight; ++i) {
            resources[j].texture[i] = std::make_unique<TextureVideo>(width, height, m_vulkanDevice,
                                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                     format);
        }
    }
}

void ImageView::updateTexture(uint32_t currentFrame, void *data, uint32_t size) {

    //auto *dataPtr = m_model->m_texture[currentFrame]->m_DataPtr;
    //std::memcpy(dataPtr, data, size);
    //m_model->m_texture[currentFrame]->updateTextureFromBuffer();
}

void ImageView::createDescriptors(bool useOffScreenImageRender) {
    // For each render pass instance:
    int numRenderPasses = 2; // Todo make adjustable
    for (int j = 0; j < numRenderPasses; ++j) {
        // 1 Create ONE descriptor pool
        // 2 Create descriptor layouts
        // 3 Create descriptor sets
        uint32_t uniformDescriptorCount = (2 * m_model->m_framesInFlight);
        uint32_t imageDescriptorSamplerCount = (1 * m_model->m_framesInFlight);
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         uniformDescriptorCount},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes,
                                                                                       m_model->m_framesInFlight);
        CHECK_RESULT(vkCreateDescriptorPool(m_vulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                            &m_model->resources[j].descriptorPool))

        for (uint32_t i = 0; i < m_model->m_framesInFlight; ++i) {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };

            VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                    setLayoutBindings.data(),
                    static_cast<uint32_t>(setLayoutBindings.size()));

            CHECK_RESULT(vkCreateDescriptorSetLayout(m_vulkanDevice->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                                     &m_model->resources[j].descriptorSetLayout[i]))

            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = m_model->resources[j].descriptorPool;
            descriptorSetAllocInfo.pSetLayouts = &m_model->resources[j].descriptorSetLayout[i];
            descriptorSetAllocInfo.descriptorSetCount = 1;
            CHECK_RESULT(vkAllocateDescriptorSets(m_vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                  &m_model->resources[j].descriptors[i]));


            std::vector<VkWriteDescriptorSet> writeDescriptorSets(3);
            writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[0].descriptorCount = 1;
            writeDescriptorSets[0].dstSet = m_model->resources[j].descriptors[i];
            writeDescriptorSets[0].dstBinding = 0;
            // TODO Make indices readable [j][i] is very confusing
            writeDescriptorSets[0].pBufferInfo = &m_renderUtils->uboDevice[j][i].bufferOne.m_DescriptorBufferInfo;

            writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[1].descriptorCount = 1;
            writeDescriptorSets[1].dstSet = m_model->resources[j].descriptors[i];
            writeDescriptorSets[1].dstBinding = 1;
            writeDescriptorSets[1].pBufferInfo = &m_renderUtils->uboDevice[j][i].bufferTwo.m_DescriptorBufferInfo;

            if (useOffScreenImageRender) {
                writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[2].descriptorCount = 1;
                writeDescriptorSets[2].dstSet = m_model->resources[j].descriptors[i];
                writeDescriptorSets[2].dstBinding = 2;
                writeDescriptorSets[2].pImageInfo = &(*m_renderUtils->secondaryRenderPasses)[0].imageInfo;
            } else {


                writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[2].descriptorCount = 1;
                writeDescriptorSets[2].dstSet = m_model->resources[j].descriptors[i];
                writeDescriptorSets[2].dstBinding = 2;
                writeDescriptorSets[2].pImageInfo = &m_model->resources[j].texture[i]->m_Descriptor;
            }
            vkUpdateDescriptorSets(m_vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                   writeDescriptorSets.data(), 0, nullptr);

        }
    }
}

void ImageView::createGraphicsPipeline() {
    int numRenderPasses = 2; // Todo make adjustable
    for (int j = 0; j < numRenderPasses; ++j) {
        for (uint32_t i = 0; i < m_model->m_framesInFlight; ++i) {

            VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(
                    &m_model->resources[j].descriptorSetLayout[i]);
            VkPushConstantRange pushconstantRanges{};
            pushconstantRanges.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            pushconstantRanges.offset = 0;
            pushconstantRanges.size = sizeof(VkRender::MousePositionPushConstant);
            info.pPushConstantRanges = &pushconstantRanges;
            info.pushConstantRangeCount = 1;
            CHECK_RESULT(
                    vkCreatePipelineLayout(m_vulkanDevice->m_LogicalDevice, &info, nullptr,
                                           &m_model->resources[j].pipelineLayout[i]))

            // Vertex bindings an attributes
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
            inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            inputAssemblyStateCI.topology = topology;

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
            depthStencilStateCI.depthTestEnable = VK_TRUE;
            depthStencilStateCI.depthWriteEnable = VK_TRUE;
            depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS;
            depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
            depthStencilStateCI.stencilTestEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo viewportStateCI{};
            viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportStateCI.viewportCount = 1;
            viewportStateCI.scissorCount = 1;

            VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
            multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampleStateCI.rasterizationSamples = m_renderUtils->msaaSamples;

            std::vector<VkDynamicState> dynamicStateEnables = {
                    VK_DYNAMIC_STATE_VIEWPORT,
                    VK_DYNAMIC_STATE_SCISSOR
            };
            VkPipelineDynamicStateCreateInfo dynamicStateCI{};
            dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
            dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());


            VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(VkRender::Vertex),
                                                                  VK_VERTEX_INPUT_RATE_VERTEX};
            std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                    {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                    {2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6},
            };
            VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
            vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputStateCI.vertexBindingDescriptionCount = 1;
            vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
            vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
            vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

            // Pipelines
            VkGraphicsPipelineCreateInfo pipelineCI{};
            pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCI.layout = m_model->resources[j].pipelineLayout[i];
            // TODO make dynamic
            if (j == 0) {
                pipelineCI.renderPass = *m_renderUtils->renderPass;
            } else {
                pipelineCI.renderPass = (*m_renderUtils->secondaryRenderPasses)[0].renderPass;
            }

            pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
            pipelineCI.pVertexInputState = &vertexInputStateCI;
            pipelineCI.pRasterizationState = &rasterizationStateCI;
            pipelineCI.pColorBlendState = &colorBlendStateCI;
            pipelineCI.pMultisampleState = &multisampleStateCI;
            pipelineCI.pViewportState = &viewportStateCI;
            pipelineCI.pDepthStencilState = &depthStencilStateCI;
            pipelineCI.pDynamicState = &dynamicStateCI;
            pipelineCI.stageCount = static_cast<uint32_t>(m_shaders->size());
            pipelineCI.pStages = m_shaders->data();

            VkResult res = vkCreateGraphicsPipelines(m_vulkanDevice->m_LogicalDevice, VK_NULL_HANDLE, 1, &pipelineCI,
                                                     nullptr,
                                                     &m_model->resources[j].pipeline[i]);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create graphics m_Pipeline");
        }
    }
}


void ImageView::draw(CommandBuffer *commandBuffer, uint32_t i) {
    if (i >= m_model->m_framesInFlight) {
        Log::Logger::getInstance()->error("Attempting to draw more buffers than available");
        return;
    }

    vkCmdBindDescriptorSets(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_model->resources[commandBuffer->renderPassIndex].pipelineLayout[i], 0,
                            1,
                            &m_model->resources[commandBuffer->renderPassIndex].descriptors[i], 0, nullptr);

    vkCmdBindPipeline(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_model->resources[commandBuffer->renderPassIndex].pipeline[i]);


    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffer->buffers[i], 0, 1, &m_model->m_mesh.vertices[i].buffer, offsets);

    if (m_model->m_mesh.indexCount > 0) {
        vkCmdBindIndexBuffer(commandBuffer->buffers[i], m_model->m_mesh.indices[i].buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer->buffers[i], m_model->m_mesh.indexCount, 1, m_model->m_mesh.firstIndex, 0, 0);
    } else {
        vkCmdDraw(commandBuffer->buffers[i], m_model->m_mesh.vertexCount, 1, 0, 0);
    }

}
