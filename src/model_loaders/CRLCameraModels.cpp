//
// Created by magnus on 3/10/22.
//

#include "CRLCameraModels.h"

#include "stb_image.h"


void CRLCameraModels::destroy(VkDevice device) {

}

void CRLCameraModels::loadFromFile(std::string filename, float scale) {

}

CRLCameraModels::Model::Model(VulkanDevice *_vulkanDevice, CRLCameraDataType type) {
    this->vulkanDevice = _vulkanDevice;
    this->modelType = type;
}

// TODO change signature to CreateMesh(), and let function decide if its device local or not
void CRLCameraModels::Model::createMesh(ArEngine::Vertex *_vertices, uint32_t vertexCount) {
    size_t vertexBufferSize = vertexCount * sizeof(ArEngine::Vertex);

    mesh.vertexCount = vertexCount;
    if (mesh.vertices.buffer == nullptr) {
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &mesh.vertices.buffer,
                &mesh.vertices.memory,
                _vertices))
    } else {
        void *data{};
        // TODO dont map and unmmap memory every time
        vkMapMemory(vulkanDevice->logicalDevice, mesh.vertices.memory, 0, vertexBufferSize, 0, &data);
        memcpy(data, _vertices, vertexBufferSize);
        vkUnmapMemory(vulkanDevice->logicalDevice, mesh.vertices.memory);
    }
}
// TODO change signature to CreateMesh(), and let function decide if its device local or not

void
CRLCameraModels::Model::createMeshDeviceLocal(ArEngine::Vertex *_vertices, uint32_t vertexCount, glm::uint32 *_indices,
                                              uint32_t
                                              indexCount) {


    size_t vertexBufferSize = vertexCount * sizeof(ArEngine::Vertex);
    size_t indexBufferSize = indexCount * sizeof(uint32_t);
    mesh.vertexCount = vertexCount;
    mesh.indexCount = indexCount;

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


void CRLCameraModels::Model::setGrayscaleTexture(crl::multisense::image::Header *streamOne) {
    if (streamOne == nullptr) {
        uint32_t size = 1920 * 1080 * 2;
        auto *p = malloc(size);
        memset(p, 0xff, size);
        textureVideo.updateTextureFromBuffer(static_cast<void *>(p), size);
        free(p);
        return;
    }

    if (streamOne->imageDataP == nullptr || streamOne->source < 1) {
        return;

    }


    if (streamOne->source == crl::multisense::Source_Disparity_Left) {


        auto *p = (uint16_t *) streamOne->imageDataP;

        // Normalize image
        uint32_t min = 100000, max = 0;
        for (int i = 0; i < streamOne->imageLength / 2; ++i) {
            uint16_t val = p[i];
            if (val > max)
                max = val;
            if (val < min)
                min = val;
        }

        for (int i = 0; i < streamOne->imageLength / 2; ++i) {
            float intermediate = (float) ((float) p[i] / (float) max) * 65536.0f;
            p[i] = (uint16_t) intermediate;
        }


        textureVideo.updateTextureFromBuffer(static_cast<void *>(p), streamOne->imageLength);

    } else {
        textureVideo.updateTextureFromBuffer(const_cast<void *>(streamOne->imageDataP), streamOne->imageLength);

    }


}


void CRLCameraModels::Model::setColorTexture(crl::multisense::image::Header *streamOne,
                                             crl::multisense::image::Header *streamTwo) {
    if (streamOne == nullptr || streamTwo == nullptr || streamOne->source < 1)
        return;

    auto *chromaBuffer = malloc(streamOne->imageLength);
    auto *lumaBuffer = malloc(streamTwo->imageLength);

    memcpy(chromaBuffer, streamOne->imageDataP, streamOne->imageLength);
    memcpy(lumaBuffer, streamTwo->imageDataP, streamTwo->imageLength);

    textureVideo.updateTextureFromBufferYUV(chromaBuffer, streamOne->imageLength, lumaBuffer,
                                            streamTwo->imageLength);


    free(chromaBuffer);
    free(lumaBuffer);

}

void CRLCameraModels::Model::setColorTexture(crl::multisense::image::Header stream) {

    auto *buf = malloc(stream.imageLength);
    memcpy(buf, stream.imageDataP, stream.imageLength);
    textureVideo.updateTextureFromBufferYUV(buf, stream.imageLength);

    free(buf);

}

// TODO REMOVE AVFRAME FROM CRLCAMERAMODELS.cpp
void CRLCameraModels::Model::setColorTexture(AVFrame *videoFrame, int bufferSize) {
    auto *yuv420pBuffer = (uint8_t *) malloc(bufferSize);
    if (videoFrame->coded_picture_number > 593)
        return;
    uint32_t size = videoFrame->width * videoFrame->height;
    memcpy(yuv420pBuffer, videoFrame->data[0], size);
    memcpy(yuv420pBuffer + (videoFrame->width * videoFrame->height), videoFrame->data[1], (videoFrame->width * videoFrame->height) / 2);

    textureVideo.updateTextureFromBufferYUV(yuv420pBuffer, bufferSize);

    free(yuv420pBuffer);

}

void
CRLCameraModels::Model::setTexture(std::basic_string<char, std::char_traits<char>, std::allocator<char>> fileName) {
    // Create texture image if not created

    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

    Texture2D texture{};

    texture.fromBuffer(pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, vulkanDevice,
                       vulkanDevice->transferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


    textureIndices.baseColor = 0;
    this->texture = texture;

}

void CRLCameraModels::Model::prepareTextureImage(uint32_t width, uint32_t height, CRLCameraDataType texType) {

    videos.height = height;
    videos.width = width;

    TextureVideo texture;
    switch (texType) {
        case CrlColorImageYUV420:
            texture = TextureVideo(videos.width, videos.height, vulkanDevice,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM);
            break;
        case CrlGrayscaleImage:
            texture = TextureVideo(videos.width, videos.height, vulkanDevice,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FORMAT_R8_UNORM);
            break;
        case CrlDisparityImage:
            texture = TextureVideo(videos.width, videos.height, vulkanDevice,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FORMAT_R16_UNORM);
            break;
        case CrlColorImage:

            break;
        case CrlPointCloud:
            texture = TextureVideo(videos.width, videos.height, vulkanDevice,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_FORMAT_R16_UNORM);
            break;
        default:
            std::cerr << "Texture type not supported yet" << std::endl;
            break;
    }
    textureVideo = texture;

}

void CRLCameraModels::draw(VkCommandBuffer commandBuffer, uint32_t i, CRLCameraModels::Model *model) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptors[i], 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &model->mesh.vertices.buffer, offsets);

    if (model->mesh.indexCount > 0) {
        vkCmdBindIndexBuffer(commandBuffer, model->mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, model->mesh.indexCount, 1, model->mesh.firstIndex, 0, 0);
    } else {
        vkCmdDraw(commandBuffer, model->mesh.vertexCount, 1, 0, 0);
    }

}


void CRLCameraModels::createDescriptors(uint32_t count, std::vector<Base::UniformBufferSet> ubo,
                                        CRLCameraModels::Model *model) {
    descriptors.resize(count);

    uint32_t uniformDescriptorCount = (3 * count);
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
        case CrlImage:
            createImageDescriptors(model, ubo);
            break;
        case CrlPointCloud:
            createPointCloudDescriptors(model, ubo);
            break;
        default:
            printf("Model type not supported yet\n");
            break;
    }

}

void CRLCameraModels::createImageDescriptors(CRLCameraModels::Model *model, std::vector<Base::UniformBufferSet> ubo) {

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
        writeDescriptorSets[1].pBufferInfo = &ubo[i].bufferTwo.descriptorBufferInfo;

        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].dstSet = descriptors[i];
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].pBufferInfo = &ubo[i].bufferThree.descriptorBufferInfo;

        if (model->textureVideo.device == nullptr) {
            writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[3].descriptorCount = 1;
            writeDescriptorSets[3].dstSet = descriptors[i];
            writeDescriptorSets[3].dstBinding = 3;
            writeDescriptorSets[3].pImageInfo = &model->texture.descriptor;
        } else {
            writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets[3].descriptorCount = 1;
            writeDescriptorSets[3].dstSet = descriptors[i];
            writeDescriptorSets[3].dstBinding = 3;
            writeDescriptorSets[3].pImageInfo = &model->textureVideo.descriptor;
        }

        vkUpdateDescriptorSets(vulkanDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }
}

void
CRLCameraModels::createPointCloudDescriptors(CRLCameraModels::Model *model, std::vector<Base::UniformBufferSet> ubo) {

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
        writeDescriptorSets[2].pImageInfo = &model->textureVideo.descriptor;

        writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[3].descriptorCount = 1;
        writeDescriptorSets[3].dstSet = descriptors[i];
        writeDescriptorSets[3].dstBinding = 3;
        writeDescriptorSets[3].pImageInfo = &model->texture.descriptor;

        vkUpdateDescriptorSets(vulkanDevice->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }

}

void CRLCameraModels::createDescriptorSetLayout(Model *pModel) {
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
    switch (pModel->modelType) {
        case CrlImage:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            break;
        case CrlPointCloud:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

            };
            break;

        default:
            printf("Model type not supported yet\n");
            break;
    }

    // ADD YCBCR SAMPLER TO DESCRIPTORS IF NEEDED
    if (pModel->textureVideo.sampler != nullptr) {
        VkDescriptorSetLayoutBinding binding{};
        setLayoutBindings[3].pImmutableSamplers = &pModel->textureVideo.sampler;
    }
    VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(setLayoutBindings.data(),
                                                                                               setLayoutBindings.size());
    CHECK_RESULT(
            vkCreateDescriptorSetLayout(vulkanDevice->logicalDevice, &layoutCreateInfo, nullptr, &descriptorSetLayout));
}

void CRLCameraModels::createPipelineLayout() {
    VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->logicalDevice, &info, nullptr, &pipelineLayout))
}

void
CRLCameraModels::createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector, ScriptType type) {
    createPipelineLayout();

    // Vertex bindings an attributes
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
    inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    VkPrimitiveTopology topology;
    if (type == ArPointCloud)
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


    VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(ArEngine::Vertex),
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

    pipelineCI.layout = pipelineLayout;

    CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->logicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));


    for (auto shaderStage: vector) {
        vkDestroyShaderModule(vulkanDevice->logicalDevice, shaderStage.module, nullptr);
    }
}


void CRLCameraModels::createRenderPipeline(const Base::RenderUtils &utils,
                                           std::vector<VkPipelineShaderStageCreateInfo> vector,
                                           Model *model,
                                           ScriptType type) {
    this->vulkanDevice = utils.device;

    createDescriptorSetLayout(model);
    createDescriptors(utils.UBCount, utils.uniformBuffers, model);

    createPipeline(*utils.renderPass, std::move(vector), type);

}
