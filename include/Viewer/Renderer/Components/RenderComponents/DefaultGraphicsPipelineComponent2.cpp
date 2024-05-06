//
// Created by magnus on 4/20/24.
//

#include "DefaultGraphicsPipelineComponent2.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"

namespace VkRender {


    bool DefaultGraphicsPipelineComponent2::cleanUp(uint32_t currentFrame, bool force) {
        pauseRendering();
        bool resourcesIdle = true;

        for (auto &data: m_renderData) {
            for (const auto &busy: data.busy) {
                if (busy.second) {
                    resourcesIdle = false;
                }
            }
        }
        if (resourcesIdle || force) {
            Log::Logger::getInstance()->trace("Cleaning up vulkan resources for DefaultGraphicsPipeline");
            for (auto &data: m_renderData) {
                vkDestroyDescriptorSetLayout(m_vulkanDevice->m_LogicalDevice, data.descriptorSetLayout, nullptr);
                vkDestroyDescriptorPool(m_vulkanDevice->m_LogicalDevice, data.descriptorPool, nullptr);

                for (const auto &pipeline: data.pipeline) {
                    vkDestroyPipeline(m_vulkanDevice->m_LogicalDevice, pipeline.second, nullptr);
                }
                for (const auto &pipeline: data.pipelineLayout) {
                    vkDestroyPipelineLayout(m_vulkanDevice->m_LogicalDevice, pipeline.second, nullptr);
                }
            }

            if (vertices.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, vertices.buffer, nullptr);
            }
            if (vertices.memory != VK_NULL_HANDLE) {
                vkFreeMemory(m_vulkanDevice->m_LogicalDevice, vertices.memory, nullptr);
            }
            if (indices.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, indices.buffer, nullptr);
            }
            if (indices.memory != VK_NULL_HANDLE) {
                vkFreeMemory(m_vulkanDevice->m_LogicalDevice, indices.memory, nullptr);
            }
            resourcesDeleted = true;
        } else {
            Log::Logger::getInstance()->trace("Waiting to clean up vulkan resources for DefaultGraphicsPipeline");
            for (auto &busy: m_renderData[currentFrame].busy) {
                busy.second = false;
            }
        }

        return resourcesIdle;
    }


    void DefaultGraphicsPipelineComponent2::update(uint32_t currentFrame) {

        /*
        for (auto &resource: resources) {
            for (size_t i = 0; i < resource.res.size(); ++i) {
                if (!resource.res[i].busy && resource.res[i].requestIdle) {
                    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, resource.res[i].pipeline, nullptr);
                    vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice,
                                            resource.res[i].pipelineLayout, nullptr);

                    setupPipeline(resource.res[i], resource.vertexShader, resource.fragmentShader,
                                  renderData[i].descriptorSetLayout, resource.renderPass, VK_SAMPLE_COUNT_64_BIT);
                }
            }
        }
*/
        if (shouldStopRendering())
            return;

        memcpy(m_renderData[currentFrame].fragShaderParamsBuffer.mapped,
               &fragShaderParams, sizeof(VkRender::ShaderValuesParams));

        memcpy(m_renderData[currentFrame].mvpBuffer.mapped,
               &mvp, sizeof(VkRender::UBOMatrix));

    }

    void DefaultGraphicsPipelineComponent2::draw(CommandBuffer *cmdBuffers) {
        uint32_t cbIndex = cmdBuffers->currentFrame;
        auto renderPassType = cmdBuffers->renderPassType;

        if (shouldStopRendering() || m_renderData[cbIndex].requestIdle[renderPassType]) {
            //resources[renderPassIndex].res[cbIndex].busy = false;
            return;
        }

        vkCmdBindPipeline(cmdBuffers->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_renderData[cbIndex].pipeline[renderPassType]);

        // TODO Make dynamic with amount of renderpassess allocated
        vkCmdBindDescriptorSets(cmdBuffers->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_renderData[cbIndex].pipelineLayout[renderPassType], 0, static_cast<uint32_t>(1),
                                &m_renderData[cbIndex].descriptorSets[cbIndex], 0, nullptr);

        // DrawCmdIndexed


        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(cmdBuffers->buffers[cmdBuffers->currentFrame], 0, 1, &vertices.buffer, offsets);
        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(cmdBuffers->buffers[cmdBuffers->currentFrame], indices.buffer, 0,
                                 VK_INDEX_TYPE_UINT32);
        }

        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdDrawIndexed(cmdBuffers->buffers[cmdBuffers->currentFrame], indices.indexCount, 1,
                             0, 0, 0);
        } else {
            vkCmdDraw(cmdBuffers->buffers[cmdBuffers->currentFrame], vertices.vertexCount, 1, 0, 0);
        }

        m_renderData[cbIndex].busy[renderPassType] = true;
    }

    void DefaultGraphicsPipelineComponent2::setupUniformBuffers() {
        for (auto &data: m_renderData) {
            m_vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                         &data.fragShaderParamsBuffer, sizeof(VkRender::ShaderValuesParams));
            m_vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                         &data.mvpBuffer, sizeof(VkRender::UBOMatrix));

            data.mvpBuffer.map();
            data.fragShaderParamsBuffer.map();
        }
    }

    void DefaultGraphicsPipelineComponent2::setupDescriptors() {
        for (auto &resource: m_renderData) {


            std::vector<VkDescriptorPoolSize> poolSizes = {
                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         m_numSwapChainImages},
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_numSwapChainImages},
            };

            VkDescriptorPoolCreateInfo descriptorPoolCI{};
            descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            descriptorPoolCI.pPoolSizes = poolSizes.data();
            descriptorPoolCI.maxSets = static_cast<uint32_t>(poolSizes.size()) * m_numSwapChainImages;
            CHECK_RESULT(
                    vkCreateDescriptorPool(m_vulkanDevice->m_LogicalDevice, &descriptorPoolCI, nullptr,
                                           &resource.descriptorPool));
            /*
                Descriptor sets
            */

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
                        vkCreateDescriptorSetLayout(m_vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI,
                                                    nullptr,
                                                    &resource.descriptorSetLayout));

                resource.descriptorSets.resize(m_numSwapChainImages);
                for (size_t i = 0; i < resource.descriptorSets.size(); i++) {
                    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                    descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    descriptorSetAllocInfo.descriptorPool = resource.descriptorPool;
                    descriptorSetAllocInfo.pSetLayouts = &resource.descriptorSetLayout;
                    descriptorSetAllocInfo.descriptorSetCount = 1;
                    CHECK_RESULT(vkAllocateDescriptorSets(m_vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                          &resource.descriptorSets[i]));

                    std::array<VkWriteDescriptorSet, 3> writeDescriptorSets{};

                    writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    writeDescriptorSets[0].descriptorCount = 1;
                    writeDescriptorSets[0].dstSet = resource.descriptorSets[i];
                    writeDescriptorSets[0].dstBinding = 0;
                    writeDescriptorSets[0].pBufferInfo = &resource.mvpBuffer.m_DescriptorBufferInfo;

                    writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    writeDescriptorSets[1].descriptorCount = 1;
                    writeDescriptorSets[1].dstSet = resource.descriptorSets[i];
                    writeDescriptorSets[1].dstBinding = 1;
                    writeDescriptorSets[1].pBufferInfo = &resource.fragShaderParamsBuffer.m_DescriptorBufferInfo;

                    writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    writeDescriptorSets[2].descriptorCount = 1;
                    writeDescriptorSets[2].dstSet = resource.descriptorSets[i];
                    writeDescriptorSets[2].dstBinding = 2;
                    writeDescriptorSets[2].pImageInfo = &m_emptyTexture.m_descriptor;

                    vkUpdateDescriptorSets(m_vulkanDevice->m_LogicalDevice,
                                           static_cast<uint32_t>(writeDescriptorSets.size()),
                                           writeDescriptorSets.data(), 0, nullptr);
                }
            }
        }

    }

    void DefaultGraphicsPipelineComponent2::setupPipeline(DefaultRenderData &data, RenderPassType type,
                                                          const std::string &vertexShader,
                                                          const std::string &fragmentShader,
                                                          VkSampleCountFlagBits sampleCountFlagBits,
                                                          VkRenderPass renderPass) {
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
        inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rasterStateCI{};
        rasterStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterStateCI.polygonMode = VK_POLYGON_MODE_FILL;
        rasterStateCI.cullMode = VK_CULL_MODE_NONE;
        rasterStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterStateCI.lineWidth = 1.0f;

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
        depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
        depthStencilStateCI.stencilTestEnable = VK_FALSE;
        depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineViewportStateCreateInfo viewportStateCI{};
        viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateCI.viewportCount = 1;
        viewportStateCI.scissorCount = 1;

        VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
        multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleStateCI.rasterizationSamples = sampleCountFlagBits;


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
                data.descriptorSetLayout
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
                vkCreatePipelineLayout(m_vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr,
                                       &data.pipelineLayout[type]));

        // Vertex bindings an attributes
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
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.layout = data.pipelineLayout[type];
        pipelineCI.renderPass = renderPass;
        pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
        pipelineCI.pVertexInputState = &vertexInputStateCI;
        pipelineCI.pRasterizationState = &rasterStateCI;
        pipelineCI.pColorBlendState = &colorBlendStateCI;
        pipelineCI.pMultisampleState = &multisampleStateCI;
        pipelineCI.pViewportState = &viewportStateCI;
        pipelineCI.pDepthStencilState = &depthStencilStateCI;
        pipelineCI.pDynamicState = &dynamicStateCI;
        pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCI.pStages = shaderStages.data();

        VkShaderModule vertModule{};
        VkShaderModule fragModule{};


        shaderStages[0] = Utils::loadShader(m_vulkanDevice->m_LogicalDevice, "spv/" + vertexShader,
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(m_vulkanDevice->m_LogicalDevice, "spv/" + fragmentShader,
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        // Default pipeline with back-face culling
        CHECK_RESULT(vkCreateGraphicsPipelines(m_vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                               &data.pipeline[type]));

        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(m_vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
        }
        // If the pipeline was updated and we had previously requested it to be idle
        data.requestIdle[type] = false;

    }

    void DefaultGraphicsPipelineComponent2::setTexture(const VkDescriptorImageInfo *info) {
        VkWriteDescriptorSet writeDescriptorSets{};

        for (const auto &data: m_renderData) {
            for (auto descriptorSet: data.descriptorSets) {
                writeDescriptorSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets.descriptorCount = 1;
                writeDescriptorSets.dstSet = descriptorSet;
                writeDescriptorSets.dstBinding = 2;
                writeDescriptorSets.pImageInfo = info;
                vkUpdateDescriptorSets(m_vulkanDevice->m_LogicalDevice, 1, &writeDescriptorSets, 0, nullptr);
            }
        }
    }

    template<>
    void
    DefaultGraphicsPipelineComponent2::bind<VkRender::OBJModelComponent>(
            VkRender::OBJModelComponent &modelComponent) {
        // Bind possible textures
        if (modelComponent.m_pixels) {
            m_objTexture.fromBuffer(modelComponent.m_pixels, modelComponent.m_texSize, VK_FORMAT_R8G8B8A8_SRGB,
                                    modelComponent.m_texWidth, modelComponent.m_texHeight, m_vulkanDevice,
                                    m_vulkanDevice->m_TransferQueue);
            stbi_image_free(modelComponent.m_pixels);

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
        CHECK_RESULT(m_vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                modelComponent.m_vertices.data()));
        // Index data
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    modelComponent.m_indices.data()));
        }

        // Create m_vulkanDevice local buffers
        // Vertex buffer
        CHECK_RESULT(m_vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &vertices.buffer,
                &vertices.memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &indices.buffer,
                    &indices.memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkBufferCopy copyRegion = {};

        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);
        }

        m_vulkanDevice->flushCommandBuffer(copyCmd, m_vulkanDevice->m_TransferQueue, true);

        vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(m_vulkanDevice->m_LogicalDevice, vertexStaging.memory, nullptr);
        if (indexBufferSize > 0) {
            vkDestroyBuffer(m_vulkanDevice->m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(m_vulkanDevice->m_LogicalDevice, indexStaging.memory, nullptr);
        }

        modelComponent.m_vertices.clear();
        modelComponent.m_indices.clear();
        boundToModel = true;

    }
};