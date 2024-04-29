/**
 * @file: MultiSense-Viewer/src/ModelLoaders/CRLCameraModels.cpp
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-10-3, mgjerde@carnegierobotics.com, Created file.
 **/

#include <stb_image.h>

#include "Viewer/ModelLoaders/CRLCameraModels.h"
#include "Viewer/Tools/Logger.h"

CRLCameraModels::Model::Model(const VkRender::RenderUtils *renderUtils, uint32_t framesInFlight) {
    this->m_VulkanDevice = renderUtils->device;
    this->m_FramesInFlight = framesInFlight;

    m_Descriptors.resize(framesInFlight);
    m_Pipeline.resize(framesInFlight);
    m_SelectionPipeline.resize(framesInFlight);
    m_TextureVideo.resize(framesInFlight);
    m_PointCloudTexture.resize(framesInFlight);
    m_TextureChromaU.resize(framesInFlight);
    m_TextureChromaV.resize(framesInFlight);
    m_ColorPointCloudBuffer.resize(framesInFlight);
    m_DescriptorSetLayout.resize(framesInFlight);
    m_PipelineLayout.resize(framesInFlight);
    m_SelectionPipelineLayout.resize(framesInFlight);

    m_Mesh.vertices.resize(framesInFlight);
    m_Mesh.indices.resize(framesInFlight);
}

CRLCameraModels::Model::~Model() {
    for (uint32_t i = 0; i < m_FramesInFlight; ++i) {
        vkFreeMemory(m_VulkanDevice->m_LogicalDevice, m_Mesh.vertices[i].memory, nullptr);
        vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, m_Mesh.vertices[i].buffer, nullptr);

        if (m_Mesh.indexCount > 0) {
            vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, m_Mesh.indices[i].buffer, nullptr);
            vkFreeMemory(m_VulkanDevice->m_LogicalDevice, m_Mesh.indices[i].memory, nullptr);
        }
        vkDestroyPipeline(m_VulkanDevice->m_LogicalDevice, m_Pipeline[i], nullptr);
        vkDestroyPipeline(m_VulkanDevice->m_LogicalDevice, m_SelectionPipeline[i], nullptr);
        vkDestroyDescriptorSetLayout(m_VulkanDevice->m_LogicalDevice, m_DescriptorSetLayout[i], nullptr);
        vkDestroyPipelineLayout(m_VulkanDevice->m_LogicalDevice, m_PipelineLayout[i], nullptr);
        vkDestroyPipelineLayout(m_VulkanDevice->m_LogicalDevice, m_SelectionPipelineLayout[i], nullptr);
    }
    vkDestroyDescriptorPool(m_VulkanDevice->m_LogicalDevice, m_DescriptorPool, nullptr);


}

void
CRLCameraModels::Model::createMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                              const std::vector<uint32_t> &indices) {
    size_t vertexBufferSize = vertices.size() * sizeof(VkRender::Vertex);
    size_t indexBufferSize = indices.size() * sizeof(uint32_t);
    m_Mesh.vertexCount = static_cast<uint32_t>(vertices.size());
    m_Mesh.indexCount = static_cast<uint32_t>(indices.size());

    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    } vertexStaging{}, indexStaging{};

    // Create staging buffers
    // Vertex m_DataPtr
    CHECK_RESULT(m_VulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            reinterpret_cast<const void *>(vertices.data())))
    // Index m_DataPtr
    if (indexBufferSize > 0) {
        CHECK_RESULT(m_VulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                reinterpret_cast<const void *>(indices.data())))
    }
    for (uint32_t i = 0; i < m_FramesInFlight; ++i) {
        // Create m_Device local buffers
        // Vertex buffer
        if (m_Mesh.vertices[i].buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, m_Mesh.vertices[i].buffer, nullptr);
            vkFreeMemory(m_VulkanDevice->m_LogicalDevice, m_Mesh.vertices[i].memory, nullptr);
            if (indexBufferSize > 0) {
                vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, m_Mesh.indices[i].buffer, nullptr);
                vkFreeMemory(m_VulkanDevice->m_LogicalDevice, m_Mesh.indices[i].memory, nullptr);
            }
        }
        CHECK_RESULT(m_VulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &m_Mesh.vertices[i].buffer,
                &m_Mesh.vertices[i].memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_VulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &m_Mesh.indices[i].buffer,
                    &m_Mesh.indices[i].memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_VulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, m_Mesh.vertices[i].buffer, 1, &copyRegion);
        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, m_Mesh.indices[i].buffer, 1, &copyRegion);
        }
        m_VulkanDevice->flushCommandBuffer(copyCmd, m_VulkanDevice->m_TransferQueue, true);
    }
    vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(m_VulkanDevice->m_LogicalDevice, vertexStaging.memory, nullptr);

    if (indexBufferSize > 0) {
        vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, indexStaging.buffer, nullptr);
        vkFreeMemory(m_VulkanDevice->m_LogicalDevice, indexStaging.memory, nullptr);
    }
}

bool CRLCameraModels::Model::updateTexture(VkRender::CRLCameraDataType type, uint32_t currentFrame) {
    switch (type) {
        case VkRender::CRL_GRAYSCALE_IMAGE:
        case VkRender::CRL_DISPARITY_IMAGE:
        case VkRender::CRL_COLOR_IMAGE_RGBA:
            m_TextureVideo[currentFrame]->updateTextureFromBuffer();
            break;
        case VkRender::CRL_COLOR_IMAGE_YUV420:
            if (m_VulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
                m_TextureVideo[currentFrame]->updateTextureFromBufferYUV();
            } else {
                m_TextureVideo[currentFrame]->updateTextureFromBuffer();
                m_TextureChromaU[currentFrame]->updateTextureFromBuffer();
                m_TextureChromaV[currentFrame]->updateTextureFromBuffer();
            }
            break;
        default:
            break;
    }

    return true;
}

bool CRLCameraModels::Model::updateTexture(VkRender::TextureData *tex, uint32_t currentFrame) {
    switch (tex->m_Type) {
        case VkRender::CRL_GRAYSCALE_IMAGE:
        case VkRender::CRL_DISPARITY_IMAGE:
        case VkRender::CRL_COLOR_IMAGE_RGBA:
            tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->updateTextureFromBuffer()
                                 : m_TextureVideo[currentFrame]->updateTextureFromBuffer();
            break;
        case VkRender::CRL_COLOR_IMAGE_YUV420:
            if (m_VulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
                tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->updateTextureFromBufferYUV()
                                     : m_TextureVideo[currentFrame]->updateTextureFromBufferYUV();
            } else {
                tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->updateTextureFromBuffer()
                                     : m_TextureVideo[currentFrame]->updateTextureFromBuffer();
                m_TextureChromaU[currentFrame]->updateTextureFromBuffer();
                m_TextureChromaV[currentFrame]->updateTextureFromBuffer();
            }
            break;
        case VkRender::CRL_CAMERA_IMAGE_NONE:
            break;
        case VkRender::CRL_DATA_NONE:
            break;
        case VkRender::CRL_POINT_CLOUD:
            break;
    }

    return true;
}

bool CRLCameraModels::Model::getTextureDataPointers(VkRender::TextureData *tex, uint32_t currentFrame) const {

    switch (tex->m_Type) {
        case VkRender::CRL_GRAYSCALE_IMAGE:
        case VkRender::CRL_DISPARITY_IMAGE:
        case VkRender::CRL_COLOR_IMAGE_RGBA:
            tex->data = tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->m_DataPtr
                                             : m_TextureVideo[currentFrame]->m_DataPtr;
            break;
        case VkRender::CRL_COLOR_IMAGE_YUV420:
            if (m_VulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
                tex->data = tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->m_DataPtr
                                                 : m_TextureVideo[currentFrame]->m_DataPtr;
                tex->data2 = tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->m_DataPtrSecondary
                                                  : m_TextureVideo[currentFrame]->m_DataPtrSecondary;
            } else {
                tex->data = tex->m_ForPointCloud ? m_PointCloudTexture[currentFrame]->m_DataPtr
                                                 : m_TextureVideo[currentFrame]->m_DataPtr;
                tex->data2 = m_TextureChromaU[currentFrame]->m_DataPtr;
                tex->data3 = m_TextureChromaV[currentFrame]->m_DataPtr;
            }
            break;
        default:
            break;
    }

    if ((tex->data2 == nullptr) && tex->m_Type == VkRender::CRL_COLOR_IMAGE_YUV420)
        return false;

    return true;
}

void CRLCameraModels::Model::createEmptyTexture(uint32_t width, uint32_t height, VkRender::CRLCameraDataType texType,
                                                bool forPointCloud, int isColorOrLuma) {
    Log::Logger::getInstance()->info("Preparing Texture m_Image {}, {}, with type {}", width, height,
                                     static_cast<int>(texType));
    VkFormat format{};
    for (uint32_t i = 0; i < m_FramesInFlight; ++i) {


        switch (texType) {
            case VkRender::CRL_COLOR_IMAGE_YUV420:
                if (m_VulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
                    format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
                } else {
                    format = VK_FORMAT_R8_UNORM;
                    m_TextureChromaU[i] = std::make_unique<TextureVideo>(width / 2, height / 2, m_VulkanDevice,
                                                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                         VK_FORMAT_R8_UNORM);
                    m_TextureChromaV[i] = std::make_unique<TextureVideo>(width / 2, height / 2, m_VulkanDevice,
                                                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                         VK_FORMAT_R8_UNORM);
                }
                break;
            case VkRender::CRL_GRAYSCALE_IMAGE:
                format = VK_FORMAT_R8_UNORM;
                break;
            case VkRender::CRL_DISPARITY_IMAGE:
                format = VK_FORMAT_R16_UNORM;
                break;
            case VkRender::CRL_COLOR_IMAGE_RGBA:
                format = VK_FORMAT_R8G8B8A8_UNORM;
                break;
            default:
                Log::Logger::getInstance()->warning("Texture type not supported yet: {}", static_cast<int>( texType));
                break;
        }

        if (forPointCloud) {
            VkFormat pointCloudFormat{};
            if (isColorOrLuma && m_VulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME))
                pointCloudFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
            else
                pointCloudFormat = VK_FORMAT_R8_UNORM;

            m_PointCloudTexture[i] = std::make_unique<TextureVideo>(width, height, m_VulkanDevice,
                                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                    pointCloudFormat);

            m_TextureChromaU[i] = std::make_unique<TextureVideo>(width / 2, height / 2, m_VulkanDevice,
                                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                 VK_FORMAT_R8_UNORM);
            m_TextureChromaV[i] = std::make_unique<TextureVideo>(width / 2, height / 2, m_VulkanDevice,
                                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                                 VK_FORMAT_R8_UNORM);
        }
        m_TextureVideo[i] = std::make_unique<TextureVideo>(width, height, m_VulkanDevice,
                                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                           format);
    }
}


void CRLCameraModels::createDescriptorSetLayout(Model *model) {
    for (uint32_t i = 0; i < model->m_FramesInFlight; ++i) {
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
        bool hasYcbcrSampler = vulkanDevice->extensionSupported(VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME);

        switch (model->m_CameraDataType) {
            case VkRender::CRL_DISPARITY_IMAGE:
                setLayoutBindings = {
                        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

                };
                break;
            case VkRender::CRL_POINT_CLOUD:

                setLayoutBindings = {
                        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},

                };
                break;

            case VkRender::CRL_COLOR_IMAGE_YUV420:
                if (!hasYcbcrSampler) {
                    setLayoutBindings = {
                            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                            {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                            {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    };
                } else {
                    setLayoutBindings = {
                            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    };
                }
                break;

            case VkRender::CRL_GRAYSCALE_IMAGE:
            case VkRender::CRL_COLOR_IMAGE_RGBA:
                setLayoutBindings = {
                        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                };

                break;
            default:
                std::cerr << "Model type not supported yet\n";
                break;
        }



        VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(),
                static_cast<uint32_t>(setLayoutBindings.size()));
        CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                                 &model->m_DescriptorSetLayout[i]))
    }
}


void
CRLCameraModels::createPipeline(VkRenderPass pT, std::vector<VkPipelineShaderStageCreateInfo> vector,
                                VkRender::CRLCameraDataType type,
                                VkPipeline *pPipelineT, VkPipelineLayout *pLayoutT,
                                VkSampleCountFlagBits samples) {

    // Vertex bindings an attributes
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
    inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    VkPrimitiveTopology topology;
    if (type == VkRender::CRL_POINT_CLOUD)
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
    multisampleStateCI.rasterizationSamples = samples;

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
    pipelineCI.layout = *pLayoutT;
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

    VkResult res = vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, VK_NULL_HANDLE, 1, &pipelineCI, nullptr,
                                             pPipelineT);
    if (res != VK_SUCCESS)
        throw std::runtime_error("Failed to create graphics m_Pipeline");


}


void CRLCameraModels::createRenderPipeline(const std::vector<VkPipelineShaderStageCreateInfo> &vector, Model *model,
                                           const VkRender::RenderUtils *renderUtils) {

    vulkanDevice = renderUtils->device;
    m_SwapChainImageCount = renderUtils->UBCount;
    m_Shaders = vector;

}

void CRLCameraModels::draw(CommandBuffer *commandBuffer, uint32_t i, Model *model, bool b) {
    if (i >= m_SwapChainImageCount)
        return;

    vkCmdBindDescriptorSets(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, model->m_PipelineLayout[i], 0,
                            1,
                            &model->m_Descriptors[i], 0, nullptr);

    vkCmdBindPipeline(commandBuffer->buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                      b ? model->m_Pipeline[i] : model->m_SelectionPipeline[i]);


    const VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffer->buffers[i], 0, 1, &model->m_Mesh.vertices[i].buffer, offsets);

    if (model->m_Mesh.indexCount > 0) {
        vkCmdBindIndexBuffer(commandBuffer->buffers[i], model->m_Mesh.indices[i].buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer->buffers[i], model->m_Mesh.indexCount, 1, model->m_Mesh.firstIndex, 0, 0);
    } else {
        vkCmdDraw(commandBuffer->buffers[i], model->m_Mesh.vertexCount, 1, 0, 0);
    }

}