//
// Created by magnus on 3/10/22.
//

#include "CRLCameraModels.h"

#include "stb_image.h"


void CRLCameraModels::destroy(VkDevice device) {

}

void CRLCameraModels::loadFromFile(std::string filename, float scale) {

}

CRLCameraModels::Model::Model(const VkRender::RenderUtils *renderUtils) {
    this->vulkanDevice = renderUtils->device;
}

CRLCameraModels::Model::~Model() {
    vkFreeMemory(vulkanDevice->logicalDevice, mesh.vertices.memory, nullptr);
    vkDestroyBuffer(vulkanDevice->logicalDevice, mesh.vertices.buffer, nullptr);

    if (mesh.indexCount > 0) {
        vkDestroyBuffer(vulkanDevice->logicalDevice, mesh.indices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, mesh.indices.memory, nullptr);
    }
}


// TODO change signature to CreateMesh(), and let function decide if its device local or not
void CRLCameraModels::Model::createMesh(VkRender::Vertex *_vertices, uint32_t vtxBufferSize) {
    mesh.vertexCount = vtxBufferSize;
    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vtxBufferSize,
            &mesh.vertices.buffer,
            &mesh.vertices.memory,
            _vertices))
}

// TODO change signature to CreateMesh(), and let function decide if its device local or not
void
CRLCameraModels::Model::createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                              const std::vector<uint32_t> &indices) {
    size_t vertexBufferSize = vertices.size() * sizeof(VkRender::Vertex);
    size_t indexBufferSize = indices.size() * sizeof(uint32_t);
    mesh.vertexCount = vertices.size();
    mesh.indexCount = indices.size();

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging{}, indexStaging{};

    // Create staging buffers
    // Vertex data
    CHECK_RESULT(vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            (void *) vertices.data()));
    // Index data
    if (indexBufferSize > 0) {
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                (void *) indices.data()));
    }

    // Create device local buffers
    // Vertex buffer
    if (mesh.vertices.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vulkanDevice->logicalDevice, mesh.vertices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, mesh.vertices.memory, nullptr);
        if (indexBufferSize > 0) {
            vkDestroyBuffer(vulkanDevice->logicalDevice, mesh.indices.buffer, nullptr);
            vkFreeMemory(vulkanDevice->logicalDevice, mesh.indices.memory, nullptr);
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
    vulkanDevice->flushCommandBuffer(copyCmd, vulkanDevice->transferQueue, true);
    vkDestroyBuffer(vulkanDevice->logicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(vulkanDevice->logicalDevice, vertexStaging.memory, nullptr);

    if (indexBufferSize > 0) {
        vkDestroyBuffer(vulkanDevice->logicalDevice, indexStaging.buffer, nullptr);
        vkFreeMemory(vulkanDevice->logicalDevice, indexStaging.memory, nullptr);
    }
}

void CRLCameraModels::Model::updateTexture(CRLCameraDataType type) {
    switch (type) {
        case AR_POINT_CLOUD:
            textureColorMap->updateTextureFromBuffer();
            break;
        case AR_GRAYSCALE_IMAGE:
        case AR_DISPARITY_IMAGE:
            textureVideo->updateTextureFromBuffer();
            break;
        case AR_COLOR_IMAGE_YUV420:
            textureVideo->updateTextureFromBufferYUV();
            break;
        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_CAMERA_IMAGE_NONE:
            break;
    }
}

void CRLCameraModels::Model::setTexture(VkRender::TextureData *tex) {

/*
     switch (AR_DISPARITY_IMAGE) {
        case AR_POINT_CLOUD:
            textureColorMap->updateTextureFromBuffer();
            break;
        case AR_GRAYSCALE_IMAGE:
        case AR_DISPARITY_IMAGE:
            textureVideo->updateTextureFromBuffer();
            break;
        case AR_COLOR_IMAGE_YUV420:
            textureVideo->updateTextureFromBufferYUV();
            break;
        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_CAMERA_IMAGE_NONE:
            break;
    }
 */

}

void
CRLCameraModels::Model::setTexture(
        const std::basic_string<char, std::char_traits<char>, std::allocator<char>> &fileName) {
    // Create texture image if not created

    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = (VkDeviceSize) texWidth * texHeight * 4;
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    textureIndices.baseColor = 0;

}

void CRLCameraModels::Model::createEmtpyTexture(uint32_t width, uint32_t height, CRLCameraDataType texType) {
    Log::Logger::getInstance()->info("Preparing Texture image {}, {}, with type {}", width, height, (int) texType);
    VkFormat format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    switch (texType) {
        case AR_COLOR_IMAGE_YUV420:
            format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
            break;
        case AR_GRAYSCALE_IMAGE:
            format = VK_FORMAT_R8_UNORM;
            break;
        case AR_DISPARITY_IMAGE:
            format = VK_FORMAT_R16_UNORM;
            break;
        case AR_POINT_CLOUD:
            format = VK_FORMAT_R16_UNORM;
            // two textures are needed for point clouds. this one for coloring and other for displacement
            textureColorMap = std::make_unique<TextureVideo>(width, height, vulkanDevice,
                                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                             VK_FORMAT_R8_UNORM);
            break;
        case AR_YUV_PLANAR_FRAME:
            format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
            break;
        default:
            std::cerr << "Texture type not supported yet" << std::endl;
            break;
    }

    textureVideo = std::make_unique<TextureVideo>(width, height, vulkanDevice,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                  format);

}

void CRLCameraModels::Model::setZoom() {


}

void CRLCameraModels::Model::getTextureDataPointer(VkRender::TextureData *tex) const {
    switch (tex->type) {
        case AR_POINT_CLOUD:
            tex->data = textureColorMap->data;
            break;
        case AR_GRAYSCALE_IMAGE:
        case AR_DISPARITY_IMAGE:
            tex->data = textureVideo->data;
            break;
        case AR_COLOR_IMAGE_YUV420:
            tex->data = textureVideo->data;
            tex->data2 = textureVideo->data2;
            break;
        case AR_YUV_PLANAR_FRAME:
            break;
        case AR_CAMERA_IMAGE_NONE:
            break;
    }
}


void CRLCameraModels::createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo,
                                        CRLCameraModels::Model *model) {
    descriptors.resize(count);

    uint32_t uniformDescriptorCount = (4 * count);
    uint32_t imageDescriptorSamplerCount = (3 * count);
    std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         uniformDescriptorCount},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

    };
    VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
    CHECK_RESULT(vkCreateDescriptorPool(vulkanDevice->logicalDevice, &poolCreateInfo, nullptr, &descriptorPool));


    /**
     * Create Descriptor Sets
     */

    switch (model->modelType) {
        case AR_DISPARITY_IMAGE:
        case AR_GRAYSCALE_IMAGE:
        case AR_COLOR_IMAGE_YUV420:
            createImageDescriptors(model, ubo);
            break;
        case AR_POINT_CLOUD:
            createPointCloudDescriptors(model, ubo);
            break;
        default:
            std::cerr << "Model type not supported yet\n";
            break;
    }

}

void
CRLCameraModels::createImageDescriptors(CRLCameraModels::Model *model, const std::vector<VkRender::UniformBufferSet> &ubo) {

    for (auto i = 0; i < ubo.size(); i++) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->logicalDevice, &descriptorSetAllocInfo, &descriptors[i]));


        std::vector<VkWriteDescriptorSet> writeDescriptorSets(model->modelType == AR_DISPARITY_IMAGE ? 5 : 4);
        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = descriptors[i];
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &ubo[i].bufferOne.descriptorBufferInfo;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = descriptors[i];
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &ubo[i].bufferTwo.descriptorBufferInfo;

        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].dstSet = descriptors[i];
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].pBufferInfo = &ubo[i].bufferThree.descriptorBufferInfo;

        if (model->textureVideo->device == nullptr) {
            writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[3].descriptorCount = 1;
            writeDescriptorSets[3].dstSet = descriptors[i];
            writeDescriptorSets[3].dstBinding = 3;
            writeDescriptorSets[3].pImageInfo = &model->texture->descriptor;
        } else {
            writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[3].descriptorCount = 1;
            writeDescriptorSets[3].dstSet = descriptors[i];
            writeDescriptorSets[3].dstBinding = 3;
            writeDescriptorSets[3].pImageInfo = &model->textureVideo->descriptor;
        }

        if (model->modelType == AR_DISPARITY_IMAGE) {
            writeDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[4].descriptorCount = 1;
            writeDescriptorSets[4].dstSet = descriptors[i];
            writeDescriptorSets[4].dstBinding = 4;
            writeDescriptorSets[4].pBufferInfo = &ubo[i].bufferFour.descriptorBufferInfo;
        }

        vkUpdateDescriptorSets(vulkanDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }
}

void
CRLCameraModels::createPointCloudDescriptors(CRLCameraModels::Model *model,
                                             const std::vector<VkRender::UniformBufferSet> &ubo) {

    for (auto i = 0; i < ubo.size(); i++) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->logicalDevice, &descriptorSetAllocInfo, &descriptors[i]));


        std::vector<VkWriteDescriptorSet> writeDescriptorSets(4);
        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = descriptors[i];
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &ubo[i].bufferOne.descriptorBufferInfo;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = descriptors[i];
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &ubo[i].bufferThree.descriptorBufferInfo;

        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].dstSet = descriptors[i];
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].pImageInfo = &model->textureVideo->descriptor;

        writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[3].descriptorCount = 1;
        writeDescriptorSets[3].dstSet = descriptors[i];
        writeDescriptorSets[3].dstBinding = 3;
        writeDescriptorSets[3].pImageInfo = &model->textureColorMap->descriptor;

        vkUpdateDescriptorSets(vulkanDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }

}

void CRLCameraModels::createDescriptorSetLayout(Model *pModel) {
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
    switch (pModel->modelType) {
        case AR_DISPARITY_IMAGE:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

            };
            break;
        case AR_POINT_CLOUD:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

            };
            break;
        case AR_YUV_PLANAR_FRAME:
        case AR_COLOR_IMAGE_YUV420:
        case AR_GRAYSCALE_IMAGE:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

            };
            break;
        default:
            std::cerr << "Model type not supported yet\n";
            break;
    }

    // ADD YCBCR SAMPLER TO DESCRIPTORS IF NEEDED
    if (nullptr != pModel->textureVideo->sampler) {
        setLayoutBindings[3].pImmutableSamplers = &pModel->textureVideo->sampler;
    }
    VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(setLayoutBindings.data(),
                                                                                               setLayoutBindings.size());
    CHECK_RESULT(
            vkCreateDescriptorSetLayout(vulkanDevice->logicalDevice, &layoutCreateInfo, nullptr, &descriptorSetLayout));
}

void CRLCameraModels::createPipelineLayout(VkPipelineLayout *pT) {
    VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

    VkPushConstantRange pushconstantRanges{};

    pushconstantRanges.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushconstantRanges.offset = 0;
    pushconstantRanges.size = sizeof(VkRender::MousePositionPushConstant);

    info.pPushConstantRanges = &pushconstantRanges;
    info.pushConstantRangeCount = 1;

    CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->logicalDevice, &info, nullptr, pT))
}

void
CRLCameraModels::createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, CRLCameraDataType type,
                                VkPipeline *pPipelineT, VkPipelineLayout *pLayoutT) {
    createPipelineLayout(pLayoutT);

    // Vertex bindings an attributes
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
    inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    VkPrimitiveTopology topology;
    if (type == AR_POINT_CLOUD)
        topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    else
        topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    inputAssemblyStateCI.topology = topology;

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
    pipelineCI.stageCount = static_cast<uint32_t>(vector.size());
    pipelineCI.pStages = vector.data();
    multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->logicalDevice, nullptr, 1, &pipelineCI, nullptr, pPipelineT));


}


void CRLCameraModels::createRenderPipeline(const std::vector<VkPipelineShaderStageCreateInfo> &vector, Model *model,
                                           const VkRender::RenderUtils *renderUtils) {

    this->vulkanDevice = renderUtils->device;

    if (initializedPipeline) {
        vkDestroyDescriptorSetLayout(vulkanDevice->logicalDevice, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanDevice->logicalDevice, descriptorPool, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->logicalDevice, pipelineLayout, nullptr);
        vkDestroyPipeline(vulkanDevice->logicalDevice, pipeline, nullptr);
        vkDestroyPipeline(vulkanDevice->logicalDevice, selectionPipeline, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->logicalDevice, selectionPipelineLayout, nullptr);
    }

    createDescriptorSetLayout(model);
    createDescriptors(renderUtils->UBCount, renderUtils->uniformBuffers, model);
    createPipeline(*renderUtils->renderPass, vector, model->modelType, &pipeline, &pipelineLayout);

    // Create selection pipeline as well
    createPipeline(renderUtils->picking->renderPass, vector, model->modelType, &selectionPipeline, &selectionPipelineLayout);
    initializedPipeline = true;
}

void CRLCameraModels::draw(VkCommandBuffer commandBuffer, uint32_t i, Model *model, bool b) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptors[i], 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, b ? pipeline : selectionPipeline);


    VkRender::MousePositionPushConstant constants{};
    constants.position = glm::vec2(640, 360);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(VkRender::MousePositionPushConstant), &constants);


    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &model->mesh.vertices.buffer, offsets);

    if (model->mesh.indexCount > 0) {
        vkCmdBindIndexBuffer(commandBuffer, model->mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, model->mesh.indexCount, 1, model->mesh.firstIndex, 0, 0);
    } else {
        vkCmdDraw(commandBuffer, model->mesh.vertexCount, 1, 0, 0);
    }

}
