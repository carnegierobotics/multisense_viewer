//
// Created by magnus on 2/20/22.
//

#include "PointCloudModel.h"


void PointCloudModel::destroy(VkDevice device) {

}

void PointCloudModel::loadFromFile(std::string filename, float scale) {

}

void PointCloudModel::transferData(Model::Vertex *_vertices, uint32_t vertexCount, glm::uint32 *_indices,
                             uint32_t
                             indexCount){
    size_t vertexBufferSize = vertexCount * sizeof(Model::Vertex);
    size_t indexBufferSize = indexCount * sizeof(uint32_t);
    model.indices.count = indexCount;
    model.vertices.count = vertexCount;

    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &model.vertices.buffer,
            &model.vertices.memory,
            _vertices));

}

void PointCloudModel::transferDataStaging(Model::Vertex *_vertices, uint32_t vertexCount, glm::uint32 *_indices,
                                    uint32_t
                                    indexCount) {

    size_t vertexBufferSize = vertexCount * sizeof(Model::Vertex);
    size_t indexBufferSize = indexCount * sizeof(uint32_t);
    model.indices.count = indexCount;
    model.vertices.count = vertexCount;
    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging, indexStaging{};

    // Create staging buffers
    // Vertex data
    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            _vertices));
    // Index data
    if (indexBufferSize > 0) {
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                _indices));
    }

    // Create device local buffers
    // Vertex buffer
    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBufferSize,
            &model.vertices.buffer,
            &model.vertices.memory));
    // Index buffer
    if (indexBufferSize > 0) {
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                indexBufferSize,
                &model.indices.buffer,
                &model.indices.memory));
    }

    // Copy from staging buffers
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = vertexBufferSize;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, model.vertices.buffer, 1, &copyRegion);
    if (indexBufferSize > 0) {
        copyRegion.size = indexBufferSize;
        vkCmdCopyBuffer(copyCmd, indexStaging.buffer, model.indices.buffer, 1, &copyRegion);
    }
    vulkanDevice->flushCommandBuffer(copyCmd, vulkanDevice->transferQueue, true);
    vkDestroyBuffer(vulkanDevice->logicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(vulkanDevice->logicalDevice, vertexStaging.memory, nullptr);

    if (indexBufferSize > 0) {
        vkDestroyBuffer(vulkanDevice->logicalDevice, indexStaging.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, indexStaging.memory, nullptr);
    }
}

void PointCloudModel::draw(VkCommandBuffer commandBuffer, uint32_t i) {

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptors[i], 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &model.vertices.buffer, offsets);

    if (model.indices.count > 0) {
        vkCmdBindIndexBuffer(commandBuffer, model.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, model.indices.count, 1, 0, 0, 0);
    } else {
        vkCmdDraw(commandBuffer, model.vertices.count, 1, 0, 0);
    }


}


void PointCloudModel::createDescriptors(uint32_t count, std::vector<Base::UniformBufferSet> ubo) {
    descriptors.resize(count);

    std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count * 3}
    };
    VkDescriptorPoolCreateInfo descriptorPoolCI{};
    descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCI.poolSizeCount = 1;
    descriptorPoolCI.pPoolSizes = poolSizes.data();
    descriptorPoolCI.maxSets = count;
    CHECK_RESULT(vkCreateDescriptorPool(vulkanDevice->logicalDevice, &descriptorPoolCI, nullptr, &descriptorPool));


    for (uint32_t i = 0; i < ubo.size(); ++i) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo = Populate::descriptorSetAllocateInfo(descriptorPool,
                                                                                                 &descriptorSetLayout,
                                                                                                 1);

        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->logicalDevice, &descriptorSetAllocInfo, &descriptors[i]));

        std::array<VkWriteDescriptorSet, 3> writeDescriptorSet = {Populate::writeDescriptorSet(descriptors[i],
                                                                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                                                               0,
                                                                                               &ubo[i]
                                                                                                       .bufferOne.descriptorBufferInfo,
                                                                                               1),

                                                                  Populate::writeDescriptorSet(descriptors[i],
                                                                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                                                               1,
                                                                                               &ubo[i]
                                                                                                       .bufferTwo
                                                                                                       .descriptorBufferInfo,
                                                                                               1),


                                                                  Populate::writeDescriptorSet(descriptors[i],
                                                                                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                                                               2,
                                                                                               &ubo[i]
                                                                                                       .bufferTwo
                                                                                                       .descriptorBufferInfo,
                                                                                               1)


        };
        vkUpdateDescriptorSets(vulkanDevice->logicalDevice, writeDescriptorSet.size(), writeDescriptorSet.data(), 0,
                               nullptr);
    }
}

void PointCloudModel::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings = {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                                                  nullptr},
                                                          {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                  nullptr},
                                                          {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                  nullptr}};
    VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(bindings.data(),
                                                                                               bindings.size());

    CHECK_RESULT(
            vkCreateDescriptorSetLayout(vulkanDevice->logicalDevice, &layoutCreateInfo, nullptr, &descriptorSetLayout));

}

void PointCloudModel::createPipelineLayout() {
    VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->logicalDevice, &info, nullptr, &pipelineLayout))
}

void PointCloudModel::createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> shaderStages) {
    createPipelineLayout();

    // Vertex bindings an attributes
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
    inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
    rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
    rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationStateCI.lineWidth = 1.0f;

    VkPipelineColorBlendAttachmentState blendAttachmentState{};
    blendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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


    std::vector<VkDynamicState> dynamicStateEnables = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicStateCI{};
    dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
    dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());


    VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(Model::Vertex),
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
    pipelineCI.layout = pipelineLayout;
    pipelineCI.renderPass = pT;
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
    multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    pipelineCI.layout = pipelineLayout;

    CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->logicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));


    for (auto shaderStage: shaderStages) {
        vkDestroyShaderModule(vulkanDevice->logicalDevice, shaderStage.module, nullptr);
    }
}


void
PointCloudModel::createRenderPipeline(const Base::RenderUtils &utils, std::vector<VkPipelineShaderStageCreateInfo> shaders) {
    this->vulkanDevice = utils.device;

    createDescriptorSetLayout();
    createDescriptors(utils.UBCount, utils.uniformBuffers);

    createPipeline(*utils.renderPass, std::move(shaders));

}
