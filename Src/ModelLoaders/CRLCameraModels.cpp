//
// Created by magnus on 3/10/22.
//

#include "CRLCameraModels.h"

#include "stb_image.h"
#include "MultiSense/Src/Tools/Logger.h"


CRLCameraModels::Model::Model(const VkRender::RenderUtils *renderUtils) {
    this->vulkanDevice = renderUtils->device;
}

CRLCameraModels::Model::~Model() {
    vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh.vertices.memory, nullptr);
    vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh.vertices.buffer, nullptr);

    if (mesh.indexCount > 0) {
        vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh.indices.buffer, nullptr);
        vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh.indices.memory, nullptr);
    }

    vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
    vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);
    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, selectionPipeline, nullptr);
    vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, selectionPipelineLayout, nullptr);

}

void
CRLCameraModels::Model::createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
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
            (void *) vertices.data()));
    // Index m_DataPtr
    if (indexBufferSize > 0) {
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                (void *) indices.data()));
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

bool CRLCameraModels::Model::updateTexture(CRLCameraDataType type) {

    /*
    if (textureVideo->m_Format == VK_FORMAT_R8G8B8A8_UNORM && type != AR_COLOR_IMAGE) {
        Log::Logger::getInstance()->error(
                "updateTexture: Format between Vulkan Texture and Expected MultiSense Texture mismatch: {} and {}",
                (int) textureVideo->m_Format, (int) type);
        return false;
    }
     */

    switch (type) {
        case AR_POINT_CLOUD:
            textureColorMap->updateTextureFromBuffer();
            break;
        case AR_GRAYSCALE_IMAGE:
        case AR_DISPARITY_IMAGE:
        case AR_COLOR_IMAGE:
            textureVideo->updateTextureFromBuffer();
            break;
        case AR_COLOR_IMAGE_YUV420:
            if (vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
                textureVideo->updateTextureFromBuffer();
                textureChromaU->updateTextureFromBuffer();
                textureChromaV->updateTextureFromBuffer();

            } else {
                textureVideo->updateTextureFromBufferYUV();

            }
            break;
        case AR_YUV_PLANAR_FRAME:
        case AR_CAMERA_IMAGE_NONE:
            break;
    }
    return true;
}


void CRLCameraModels::Model::createEmptyTexture(uint32_t width, uint32_t height, CRLCameraDataType texType) {
    Log::Logger::getInstance()->info("Preparing Texture m_Image {}, {}, with type {}", width, height, (int) texType);
    VkFormat format{};
    switch (texType) {
        case AR_COLOR_IMAGE_YUV420:
            if (!vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME))
                format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
            else {
                format = VK_FORMAT_R8_UNORM;
                textureChromaU = std::make_unique<TextureVideo>(width, height, vulkanDevice,
                                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                VK_FORMAT_R8_UNORM);
                textureChromaV = std::make_unique<TextureVideo>(width, height, vulkanDevice,
                                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                VK_FORMAT_R8_UNORM);
            }
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
        case AR_COLOR_IMAGE:
            format = VK_FORMAT_R8G8B8A8_UNORM;
            break;
        default:
            std::cerr << "Texture type not supported yet" << std::endl;
            break;
    }

    textureVideo = std::make_unique<TextureVideo>(width, height, vulkanDevice,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                  format);

}


bool CRLCameraModels::Model::getTextureDataPointers(VkRender::TextureData *tex) const {

    switch (tex->m_Type) {
        case AR_POINT_CLOUD:
            tex->data = textureColorMap->m_DataPtr;
            break;
        case AR_GRAYSCALE_IMAGE:
        case AR_DISPARITY_IMAGE:
        case AR_COLOR_IMAGE:
            tex->data = textureVideo->m_DataPtr;

            break;
        case AR_COLOR_IMAGE_YUV420:
            tex->data = textureVideo->m_DataPtr;
            tex->data2 = textureVideo->m_DataPtrSecondary;
            if (vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
                tex->data2 = textureChromaU->m_DataPtr;
                tex->data2 = textureChromaV->m_DataPtr;
            }
            break;
        case AR_YUV_PLANAR_FRAME:
        case AR_CAMERA_IMAGE_NONE:
            break;
    }

    if (tex->data2 == nullptr && tex->m_Type == AR_COLOR_IMAGE_YUV420)
        return false;

    return true;
}


void CRLCameraModels::createDescriptors(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo,
                                        CRLCameraModels::Model *model) {
    model->descriptors.resize(count);

    uint32_t uniformDescriptorCount = (4 * count);
    uint32_t imageDescriptorSamplerCount = (3 * count);
    std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         uniformDescriptorCount},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

    };
    VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
    CHECK_RESULT(
            vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr, &model->descriptorPool));


    /**
     * Create Descriptor Sets
     */

    switch (model->modelType) {
        case AR_DISPARITY_IMAGE:
        case AR_GRAYSCALE_IMAGE:
        case AR_COLOR_IMAGE_YUV420:
        case AR_COLOR_IMAGE:
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
CRLCameraModels::createImageDescriptors(CRLCameraModels::Model *model,
                                        const std::vector<VkRender::UniformBufferSet> &ubo) {

    for (size_t i = 0; i < ubo.size(); i++) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = model->descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &model->descriptorSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              &model->descriptors[i]));


        bool hasYcbcrSampler = vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);

        std::vector<VkWriteDescriptorSet> writeDescriptorSets(2);
        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = model->descriptors[i];
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &ubo[i].bufferOne.m_DescriptorBufferInfo;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = model->descriptors[i];
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &ubo[i].bufferTwo.m_DescriptorBufferInfo;

        if (hasYcbcrSampler) {
            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = model->descriptors[i];
            writeDescriptorSet.dstBinding = 2;
            writeDescriptorSet.pImageInfo = &model->textureVideo->m_Descriptor;
            writeDescriptorSets.emplace_back(writeDescriptorSet);

        } else {
            VkWriteDescriptorSet writeDescriptorSetLuma{};
            VkWriteDescriptorSet writeDescriptorSetChromaU{};
            VkWriteDescriptorSet writeDescriptorSetChromaV{};

            writeDescriptorSetLuma.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSetLuma.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSetLuma.descriptorCount = 1;
            writeDescriptorSetLuma.dstSet = model->descriptors[i];
            writeDescriptorSetLuma.dstBinding = 2;
            writeDescriptorSetLuma.pImageInfo = &model->textureVideo->m_Descriptor;

            writeDescriptorSetChromaU.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSetChromaU.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSetChromaU.descriptorCount = 1;
            writeDescriptorSetChromaU.dstSet = model->descriptors[i];
            writeDescriptorSetChromaU.dstBinding = 3;
            writeDescriptorSetChromaU.pImageInfo = &model->textureChromaU->m_Descriptor;

            writeDescriptorSetChromaV.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSetChromaV.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSetChromaV.descriptorCount = 1;
            writeDescriptorSetChromaV.dstSet = model->descriptors[i];
            writeDescriptorSetChromaV.dstBinding = 4;
            writeDescriptorSetChromaV.pImageInfo = &model->textureChromaV->m_Descriptor;

            writeDescriptorSets.emplace_back(writeDescriptorSetLuma);
            writeDescriptorSets.emplace_back(writeDescriptorSetChromaU);
            writeDescriptorSets.emplace_back(writeDescriptorSetChromaV);
        }

        vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }
}

void
CRLCameraModels::createPointCloudDescriptors(CRLCameraModels::Model *model,
                                             const std::vector<VkRender::UniformBufferSet> &ubo) {

    for (size_t i = 0; i < ubo.size(); i++) {

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = model->descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = &model->descriptorSetLayout;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              &model->descriptors[i]));


        std::vector<VkWriteDescriptorSet> writeDescriptorSets(4);
        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].dstSet = model->descriptors[i];
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].pBufferInfo = &ubo[i].bufferOne.m_DescriptorBufferInfo;

        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].dstSet = model->descriptors[i];
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].pBufferInfo = &ubo[i].bufferThree.m_DescriptorBufferInfo;

        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].dstSet = model->descriptors[i];
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].pImageInfo = &model->textureVideo->m_Descriptor;

        writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[3].descriptorCount = 1;
        writeDescriptorSets[3].dstSet = model->descriptors[i];
        writeDescriptorSets[3].dstBinding = 3;
        writeDescriptorSets[3].pImageInfo = &model->textureColorMap->m_Descriptor;

        vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, NULL);
    }

}

void CRLCameraModels::createDescriptorSetLayout(Model *model) {
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
    bool hasYcbcrSampler = !vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);

    switch (model->modelType) {
        case AR_DISPARITY_IMAGE:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

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
            if (!hasYcbcrSampler) {
                setLayoutBindings = {
                        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                };
                break;
            }
        case AR_GRAYSCALE_IMAGE:
        case AR_COLOR_IMAGE:
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };

            break;
        default:
            std::cerr << "Model type not supported yet\n";
            break;
    };


// ADD YCBCR SAMPLER TO DESCRIPTORS IF NEEDED
    if (nullptr != model->textureVideo->m_Sampler) {
        setLayoutBindings[2].pImmutableSamplers = &model->textureVideo->m_Sampler;
    }
    VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(setLayoutBindings.data(),
                                                                                               static_cast<uint32_t>(setLayoutBindings.size()));
    CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &layoutCreateInfo, nullptr, &model->descriptorSetLayout));
}

void CRLCameraModels::createPipelineLayout(VkPipelineLayout *pT, const VkDescriptorSetLayout &layout) {
    VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&layout, 1);

    VkPushConstantRange pushconstantRanges{};

    pushconstantRanges.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushconstantRanges.offset = 0;
    pushconstantRanges.size = sizeof(VkRender::MousePositionPushConstant);

    info.pPushConstantRanges = &pushconstantRanges;
    info.pushConstantRangeCount = 1;

    CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &info, nullptr, pT))
}

void
CRLCameraModels::createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector,
                                CRLCameraDataType type,
                                VkPipeline *pPipelineT, VkPipelineLayout *pLayoutT, Model *model) {
    createPipelineLayout(pLayoutT, model->descriptorSetLayout);

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
    pipelineCI.layout = model->pipelineLayout;
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

    VkResult res = vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                             pPipelineT);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create graphics pipeline");


}


void CRLCameraModels::createRenderPipeline(const std::vector<VkPipelineShaderStageCreateInfo> &vector, Model *model,
                                           const VkRender::RenderUtils *renderUtils) {

    this->vulkanDevice = renderUtils->device;

    if (model->initializedPipeline) {
        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, model->descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, model->descriptorPool, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, model->pipelineLayout, nullptr);
        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, model->pipeline, nullptr);
        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, model->selectionPipeline, nullptr);
        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, model->selectionPipelineLayout, nullptr);
    }

    createDescriptorSetLayout(model);
    createDescriptors(renderUtils->UBCount, renderUtils->uniformBuffers, model);
    createPipeline(*renderUtils->renderPass, vector, model->modelType, &model->pipeline, &model->pipelineLayout,
                   model);

    // Create selection pipeline as well
    createPipeline(renderUtils->picking->renderPass, vector, model->modelType, &model->selectionPipeline,
                   &model->selectionPipelineLayout, model);
    model->initializedPipeline = true;
}

void CRLCameraModels::draw(VkCommandBuffer commandBuffer, uint32_t i, Model *model, bool b) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, model->pipelineLayout, 0, 1,
                            &model->descriptors[i], 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, b ? model->pipeline : model->selectionPipeline);


    VkRender::MousePositionPushConstant constants{};
    constants.position = glm::vec2(640, 360);
    vkCmdPushConstants(commandBuffer, model->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
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
