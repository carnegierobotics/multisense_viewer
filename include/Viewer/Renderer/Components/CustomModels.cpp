//
// Created by magnus on 10/3/23.
//

#include "Viewer/Renderer/Components/CustomModels.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"

namespace VkRender {
    CustomModelComponent::Model::Model(const VkRender::RenderUtils *renderUtils) {
        this->vulkanDevice = renderUtils->device;

        VkRender::ScriptUtils::ImageData imgData{};
        uploadMeshDeviceLocal(imgData.quad.vertices, imgData.quad.indices);

    }

    CustomModelComponent::Model::~Model() {
        vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh.vertices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh.vertices.memory, nullptr);

        if (mesh.indexCount > 0) {
            vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh.indices.buffer, nullptr);
            vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh.indices.memory, nullptr);
        }
    }


    void CustomModelComponent::Model::uploadMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                                            const std::vector<uint32_t> &indices) {
        size_t vertexBufferSize = vertices.size() * sizeof(VkRender::Vertex);
        size_t indexBufferSize = indices.size() * sizeof(uint32_t);
        mesh.vertexCount = static_cast<uint32_t>(vertices.size());
        mesh.indexCount = static_cast<uint32_t>(indices.size());

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex m_DataPtr
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                reinterpret_cast<const void *>(vertices.data())))
        // Index m_DataPtr
        if (indexBufferSize > 0) {
            CHECK_RESULT(vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    reinterpret_cast<const void *>(indices.data())))
        }
        // Create m_Device local buffers
        // Vertex buffer
        if (mesh.vertices.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh.vertices.buffer, nullptr);
            vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh.vertices.memory, nullptr);
            if (indexBufferSize > 0) {
                vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh.indices.buffer, nullptr);
                vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh.indices.memory, nullptr);
            }
        }
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &mesh.vertices.buffer,
                &mesh.vertices.memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &mesh.indices.buffer,
                    &mesh.indices.memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, mesh.vertices.buffer, 1, &copyRegion);
        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, mesh.indices.buffer, 1, &copyRegion);
        }
        vulkanDevice->flushCommandBuffer(copyCmd, vulkanDevice->m_TransferQueue, true);
        vkDestroyBuffer(vulkanDevice->m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(vulkanDevice->m_LogicalDevice, vertexStaging.memory, nullptr);

        if (indexBufferSize > 0) {
            vkDestroyBuffer(vulkanDevice->m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(vulkanDevice->m_LogicalDevice, indexStaging.memory, nullptr);
        }
    }

    void CustomModelComponent::createDescriptorSetLayout() {
        for (auto &descriptorSetLayout: descriptorSetLayouts) {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                    setLayoutBindings.data(),
                    static_cast<uint32_t>(setLayoutBindings.size()));
            CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                                     &descriptorSetLayout));
        }
    }

    void CustomModelComponent::createDescriptorPool() {
        for (auto &descriptorPool: descriptorPools) {
            uint32_t uniformDescriptorCount = renderer->UBCount;
            std::vector<VkDescriptorPoolSize> poolSizes = {
                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformDescriptorCount},
            };
            VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes,
                                                                                           renderer->UBCount);

            CHECK_RESULT(
                    vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));
        }
    }

    void CustomModelComponent::createDescriptorSets() {
        descriptors.resize(renderer->UBCount);

        for (size_t i = 0; i < descriptorSetLayouts.size(); ++i) {
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = descriptorPools[i];
            descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayouts[i];
            descriptorSetAllocInfo.descriptorSetCount = 1;
            CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                  &descriptors[i]));

            std::vector<VkWriteDescriptorSet> writeDescriptorSets(1);
            writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[0].descriptorCount = 1;
            writeDescriptorSets[0].dstSet = descriptors[i];
            writeDescriptorSets[0].dstBinding = 0;
            writeDescriptorSets[0].pBufferInfo = &UBOBuffers[i].m_DescriptorBufferInfo;


            vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                   writeDescriptorSets.data(), 0, nullptr);
        }
    }

    void CustomModelComponent::createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> vector) {
        for (size_t i = 0; i < pipelineLayouts.size(); ++i) {


            VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&descriptorSetLayouts[i], 1);
            CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &info, nullptr, &pipelineLayouts[i]))

            // Vertex bindings an attributes
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
            inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
            rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
            rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterizationStateCI.lineWidth = 1.0f;

            // Enable blending
            VkPipelineColorBlendAttachmentState blendAttachmentState{};
            blendAttachmentState.blendEnable = VK_TRUE;
            blendAttachmentState.colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                    VK_COLOR_COMPONENT_A_BIT;
            blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
            blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

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
            multisampleStateCI.rasterizationSamples = renderer->msaaSamples;


            std::vector<VkDynamicState> dynamicStateEnables = {
                    VK_DYNAMIC_STATE_VIEWPORT,
                    VK_DYNAMIC_STATE_SCISSOR
            };
            VkPipelineDynamicStateCreateInfo dynamicStateCI{};
            dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
            dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());


            VkVertexInputBindingDescription vertexInputBinding = {
                    0, sizeof(VkRender::Vertex),
                    VK_VERTEX_INPUT_RATE_VERTEX
            };
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
            pipelineCI.layout = pipelineLayouts[i];
            pipelineCI.renderPass = *renderer->renderPass;
            pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
            pipelineCI.pVertexInputState = &vertexInputStateCI;
            pipelineCI.pRasterizationState = &rasterizationStateCI;
            pipelineCI.pColorBlendState = &colorBlendStateCI;
            pipelineCI.pMultisampleState = &multisampleStateCI;
            pipelineCI.pViewportState = &viewportStateCI;
            pipelineCI.pDepthStencilState = &depthStencilStateCI;
            pipelineCI.pDynamicState = &dynamicStateCI;
            pipelineCI.stageCount = static_cast<uint32_t>(vector.size());
            pipelineCI.pStages = vector.data();

            VkResult res = vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                                     &pipelines[i]);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create graphics m_Pipeline");
        }
    }

    void CustomModelComponent::draw(CommandBuffer *commandBuffer, uint32_t cbIndex) {
        if (cbIndex >= renderer->UBCount)
            return;
        vkCmdBindDescriptorSets(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayouts[cbIndex],
                                0, 1,
                                &descriptors[cbIndex], 0, nullptr);
        vkCmdBindPipeline(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[cbIndex]);

        const VkDeviceSize offsets[1] = {0};

        vkCmdBindVertexBuffers(commandBuffer->buffers[cbIndex], 0, 1, &model->mesh.vertices.buffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer->buffers[cbIndex], model->mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer->buffers[cbIndex], model->mesh.indexCount, 1, model->mesh.firstIndex, 0, 0);

        resourcesInUse[cbIndex] = true;
    }
};