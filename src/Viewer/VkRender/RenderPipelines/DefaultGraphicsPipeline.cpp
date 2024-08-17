//
// Created by magnus on 4/20/24.
//

#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Core/VulkanResourceManager.h"

namespace VkRender {


    DefaultGraphicsPipeline::DefaultGraphicsPipeline(Renderer &m_context,
                                                                       const RenderPassInfo &renderPassInfo,
                                                                       const std::string &vertexShader,
                                                                       const std::string &fragmentShader)
            : m_vulkanDevice(m_context.vkDevice()),
              m_renderPassInfo(std::move(renderPassInfo)) {

        m_numSwapChainImages = m_context.swapChainBuffers().size();
        m_vulkanDevice = m_context.vkDevice();

        m_emptyTexture.fromKtxFile((Utils::getTexturePath() / "empty.ktx").string(), VK_FORMAT_R8G8B8A8_UNORM,
                                   &m_vulkanDevice, m_vulkanDevice.m_TransferQueue);

        m_vertexShader = vertexShader;
        m_fragmentShader = fragmentShader;

        m_renderData.resize(m_numSwapChainImages);

        setupUniformBuffers();
        setupDescriptors();

        setupPipeline();
    }


    void DefaultGraphicsPipeline::cleanUp() {
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        VkFence fence;
        VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
        vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence);

        Indices& indices = this->indices;
        Vertices& vertices = this->vertices;
        VkDescriptorSetLayout layout =  m_sharedRenderData.descriptorSetLayout;
        VkDescriptorPool pool =  m_sharedRenderData.descriptorPool;

        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, indices, vertices, layout, pool]() {
                    vkDestroyDescriptorSetLayout(logicalDevice, layout, nullptr);
                    vkDestroyDescriptorPool(logicalDevice, pool, nullptr);

                    if (vertices.buffer != VK_NULL_HANDLE) {
                        vkDestroyBuffer(logicalDevice, vertices.buffer, nullptr);
                    }
                    if (vertices.memory != VK_NULL_HANDLE) {
                        vkFreeMemory(logicalDevice, vertices.memory, nullptr);
                    }
                    if (indices.buffer != VK_NULL_HANDLE) {
                        vkDestroyBuffer(logicalDevice, indices.buffer, nullptr);
                    }
                    if (indices.memory != VK_NULL_HANDLE) {
                        vkFreeMemory(logicalDevice, indices.memory, nullptr);
                    }
                },
                fence);


    }

    void DefaultGraphicsPipeline::setupUniformBuffers() {
        for (auto &data: m_renderData) {
            m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        &data.fragShaderParamsBuffer, sizeof(VkRender::ShaderValuesParams));
            m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        &data.mvpBuffer, sizeof(VkRender::UBOMatrix));

            data.mvpBuffer.map();
            data.fragShaderParamsBuffer.map();
        }
    }


    void DefaultGraphicsPipeline::setupDescriptors() {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         m_numSwapChainImages * 2},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_numSwapChainImages * 2},
        };

        VkDescriptorPoolCreateInfo descriptorPoolCI{};
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        descriptorPoolCI.pPoolSizes = poolSizes.data();
        descriptorPoolCI.maxSets = m_numSwapChainImages * static_cast<uint32_t>(poolSizes.size());
        CHECK_RESULT(
                vkCreateDescriptorPool(m_vulkanDevice.m_LogicalDevice, &descriptorPoolCI, nullptr,
                                       &m_sharedRenderData.descriptorPool));


        // Scene (matrices and environment maps)
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1,
                                                                      VK_SHADER_STAGE_VERTEX_BIT |
                                                                      VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            CHECK_RESULT(
                    vkCreateDescriptorSetLayout(m_vulkanDevice.m_LogicalDevice, &descriptorSetLayoutCI,
                                                nullptr,
                                                &m_sharedRenderData.descriptorSetLayout));

            for (auto &resource: m_renderData) {
                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = m_sharedRenderData.descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &m_sharedRenderData.descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                VkResult res = vkAllocateDescriptorSets(m_vulkanDevice.m_LogicalDevice, &descriptorSetAllocInfo,
                                                        &resource.descriptorSet);
                if (res != VK_SUCCESS)
                    throw std::runtime_error("Failed to allocate descriptor sets");

                std::array<VkWriteDescriptorSet, 3> writeDescriptorSets{};

                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].dstSet = resource.descriptorSet;
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].pBufferInfo = &resource.mvpBuffer.m_DescriptorBufferInfo;

                writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[1].descriptorCount = 1;
                writeDescriptorSets[1].dstSet = resource.descriptorSet;
                writeDescriptorSets[1].dstBinding = 1;
                writeDescriptorSets[1].pBufferInfo = &resource.fragShaderParamsBuffer.m_DescriptorBufferInfo;

                writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[2].descriptorCount = 1;
                writeDescriptorSets[2].dstSet = resource.descriptorSet;
                writeDescriptorSets[2].dstBinding = 2;
                writeDescriptorSets[2].pImageInfo = &m_emptyTexture.m_descriptor;

                vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice,
                                       static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, nullptr);
            }
        }
    }


    void DefaultGraphicsPipeline::setupPipeline() {

        // Vertex bindings an attributes
        VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(VkRender::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
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
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages(2);
        VkShaderModule vertModule{};
        VkShaderModule fragModule{};
        shaderStages[0] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_vertexShader,
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_fragmentShader,
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        VulkanGraphicsPipelineCreateInfo createInfo( m_renderPassInfo.renderPass, m_vulkanDevice);
        createInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                       VK_FRONT_FACE_COUNTER_CLOCKWISE);
        createInfo.msaaSamples = m_renderPassInfo.sampleCount;
        createInfo.shaders = shaderStages;
        createInfo.descriptorSetLayout = m_sharedRenderData.descriptorSetLayout;
        createInfo.vertexInputState = vertexInputStateCI;

        m_sharedRenderData.graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(createInfo);

        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(m_vulkanDevice.m_LogicalDevice, shaderStage.module, nullptr);
        }

    }

    void DefaultGraphicsPipeline::update(uint32_t currentFrame) {
        memcpy(m_renderData[currentFrame].fragShaderParamsBuffer.mapped,
               &m_fragParams, sizeof(VkRender::ShaderValuesParams));
        memcpy(m_renderData[currentFrame].mvpBuffer.mapped,
               &m_vertexParams, sizeof(VkRender::UBOMatrix));

    }
    void DefaultGraphicsPipeline::updateTransform(const TransformComponent &transform) {
        m_vertexParams.model = transform.GetTransform();

    }
    void DefaultGraphicsPipeline::updateView(const Camera &camera) {
        m_vertexParams.view = camera.matrices.view;
        m_vertexParams.projection = camera.matrices.perspective;
        m_vertexParams.camPos = camera.pose.pos;
    }


    void DefaultGraphicsPipeline::draw(CommandBuffer &cmdBuffers) {
        const uint32_t &cbIndex = *cmdBuffers.frameIndex;
        vkCmdBindPipeline(cmdBuffers.buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, m_sharedRenderData.graphicsPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmdBuffers.buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_sharedRenderData.graphicsPipeline->getPipelineLayout(), 0, static_cast<uint32_t>(1),
                                &m_renderData[cbIndex].descriptorSet, 0, nullptr);
        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(cmdBuffers.buffers[cbIndex], 0, 1, &vertices.buffer, offsets);
        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(cmdBuffers.buffers[cbIndex], indices.buffer, 0,
                                 VK_INDEX_TYPE_UINT32);
        }
        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdDrawIndexed(cmdBuffers.buffers[cbIndex], indices.indexCount, 1,
                             0, 0, 0);
        } else {
            vkCmdDraw(cmdBuffers.buffers[cbIndex], vertices.vertexCount, 1, 0, 0);
        }
    }

    void DefaultGraphicsPipeline::setTexture(const VkDescriptorImageInfo *info) {
        VkWriteDescriptorSet writeDescriptorSets{};

        for (const auto &data: m_renderData) {
            writeDescriptorSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets.descriptorCount = 1;
            writeDescriptorSets.dstSet = data.descriptorSet;
            writeDescriptorSets.dstBinding = 2;
            writeDescriptorSets.pImageInfo = info;
            vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice, 1, &writeDescriptorSets, 0, nullptr);

        }
    }

    DefaultGraphicsPipeline::~DefaultGraphicsPipeline() {
        cleanUp();
    }

    void DefaultGraphicsPipeline::bind(VkRender::MeshComponent &modelComponent) {

        // Bind possible textures
        if (modelComponent.m_pixels) {
            m_objTexture.fromBuffer(modelComponent.m_pixels, modelComponent.m_texSize, VK_FORMAT_R8G8B8A8_SRGB,
                                    modelComponent.m_texWidth, modelComponent.m_texHeight, &m_vulkanDevice,
                                    m_vulkanDevice.m_TransferQueue);
            // TODO free m_pixels once we are done with it. We are currently leaving memory leaks
            setTexture(&m_objTexture.m_descriptor);
        }
        // Bind vertex/index buffers from model
        indices.indexCount = modelComponent.m_indices.size();
        size_t vertexBufferSize = modelComponent.m_vertices.size() * sizeof(VkRender::Vertex);
        size_t indexBufferSize = modelComponent.m_indices.size() * sizeof(uint32_t);

        assert(vertexBufferSize > 0);

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex data
        CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                modelComponent.m_vertices.data()));
        // Index data
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice.createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    modelComponent.m_indices.data()));
        }

        // Create m_vulkanDevice local buffers
        // Vertex buffer
        CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &vertices.buffer,
                &vertices.memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice.createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &indices.buffer,
                    &indices.memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkBufferCopy copyRegion = {};

        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);
        }

        m_vulkanDevice.flushCommandBuffer(copyCmd, m_vulkanDevice.m_TransferQueue, true);

        vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(m_vulkanDevice.m_LogicalDevice, vertexStaging.memory, nullptr);
        if (indexBufferSize > 0) {
            vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(m_vulkanDevice.m_LogicalDevice, indexStaging.memory, nullptr);
        }

    }

};