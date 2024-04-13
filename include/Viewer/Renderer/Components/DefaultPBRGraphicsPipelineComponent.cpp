//
// Created by magnus on 4/12/24.
//
#include "Viewer/Renderer/Components/DefaultPBRGraphicsPipelineComponent.h"
#include "Viewer/Tools/Utils.h"

namespace RenderResource {


    void DefaultPBRGraphicsPipelineComponent::renderNode(CommandBuffer *commandBuffer, uint32_t cbIndex,
                                                         VkRender::Node *node,
                                                         VkRender::Material::AlphaMode alphaMode) {
        if (node->mesh) {
            // Render mesh primitives
            for (VkRender::Primitive *primitive: node->mesh->primitives) {
                if (primitive->material.alphaMode == alphaMode) {
                    std::string pipelineName = "pbr";
                    std::string pipelineVariant = "";

                    if (primitive->material.unlit) {
                        // KHR_materials_unlit
                        pipelineName = "unlit";
                    };

                    // Material properties define if we e.g. need to bind a pipeline variant with culling disabled (double sided)
                    if (alphaMode == VkRender::Material::ALPHAMODE_BLEND) {
                        pipelineVariant = "_alpha_blending";
                    } else {
                        if (primitive->material.doubleSided) {
                            pipelineVariant = "_double_sided";
                        }
                    }

                    const VkPipeline pipeline = resources[cbIndex].pipelines[pipelineName + pipelineVariant];

                    if (pipeline != resources[cbIndex].boundPipeline) {
                        vkCmdBindPipeline(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                        resources[cbIndex].boundPipeline = pipeline;
                    }

                    const std::vector<VkDescriptorSet> descriptors = {
                            resources[cbIndex].descriptorSets[cbIndex],
                            primitive->material.descriptorSet,
                            node->mesh->uniformBuffer.descriptorSet,
                            resources[cbIndex].descriptorSetMaterials
                    };
                    vkCmdBindDescriptorSets(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            resources[cbIndex].pipelineLayouts[pipelineName], 0, static_cast<uint32_t>(descriptors.size()),
                                            descriptors.data(), 0, NULL);

                    // Pass material index for this primitive using a push constant, the shader uses this to index into the material buffer
                    vkCmdPushConstants(commandBuffer->buffers[cbIndex], resources[cbIndex].pipelineLayouts[pipelineName], VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                       sizeof(uint32_t), &primitive->material.index);

                    if (primitive->hasIndices) {
                        vkCmdDrawIndexed(commandBuffer->buffers[cbIndex], primitive->indexCount, 1,
                                         primitive->firstIndex, 0, 0);
                    } else {
                        vkCmdDraw(commandBuffer->buffers[cbIndex], primitive->vertexCount, 1, 0, 0);
                    }
                }
            }

        };
        for (auto child: node->children) {
            renderNode(commandBuffer, cbIndex, child, alphaMode);
        }
    }

    void DefaultPBRGraphicsPipelineComponent::draw(CommandBuffer *commandBuffer, uint32_t cbIndex,
                                                   const VkRender::GLTFModelComponent &component) {
        VkDeviceSize offsets[1] = { 0 };

        vkCmdBindVertexBuffers(commandBuffer->buffers[cbIndex], 0, 1, &component.model->vertices.buffer, offsets);
        if (component.model->indices.buffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(commandBuffer->buffers[cbIndex], component.model->indices.buffer, 0,
                                 VK_INDEX_TYPE_UINT32);
        }

        resources[cbIndex].boundPipeline = VK_NULL_HANDLE;

        // Opaque primitives first
        for (auto node : component.model->nodes) {
            renderNode(commandBuffer, cbIndex, node, VkRender::Material::ALPHAMODE_OPAQUE);
        }
        // Alpha masked primitives
        for (auto node : component.model->nodes) {
            renderNode(commandBuffer, cbIndex, node, VkRender::Material::ALPHAMODE_MASK);
        }
        // Transparent primitives
        // TODO: Correct depth sorting
        for (auto node : component.model->nodes) {
            renderNode(commandBuffer, cbIndex, node, VkRender::Material::ALPHAMODE_BLEND);
        }
    }

    void DefaultPBRGraphicsPipelineComponent::setupUniformBuffers(Resource& resource) {
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                       &resource.bufferParams, sizeof(VkRender::ShaderValuesParams));
            vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                       &resource.bufferScene, sizeof(VkRender::UBOMatrix));

            resource.bufferScene.map();
            resource.bufferParams.map();

    }

    void DefaultPBRGraphicsPipelineComponent::update(){

        memcpy(resources[renderUtils->swapchainIndex].bufferParams.mapped, &resources[renderUtils->swapchainIndex].shaderValuesParams, sizeof(VkRender::ShaderValuesParams));
        memcpy(resources[renderUtils->swapchainIndex].bufferScene.mapped, &resources[renderUtils->swapchainIndex].uboMatrix, sizeof(VkRender::UBOMatrix));

    }

    void DefaultPBRGraphicsPipelineComponent::setupDescriptors(Resource& resource, const VkRender::GLTFModelComponent &component,
                                                               const RenderResource::SkyboxGraphicsPipelineComponent &skyboxComponent) {
/*
			Descriptor Pool
		*/
        uint32_t imageSamplerCount = 0;
        uint32_t materialCount = 0;
        uint32_t meshCount = 0;

        // Environment samplers (radiance, irradiance, brdf lut)
        imageSamplerCount += 3;

        for ([[maybe_unused]] auto &material: component.model->materials) {
            imageSamplerCount += 5;
            materialCount++;
        }
        for (auto node: component.model->linearNodes) {
            if (node->mesh) {
                meshCount++;
            }
        }


        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         (4 + meshCount) * renderUtils->UBCount},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageSamplerCount * renderUtils->UBCount},
                // One SSBO for the shader material buffer
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1}
        };
        VkDescriptorPoolCreateInfo descriptorPoolCI{};
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        descriptorPoolCI.pPoolSizes = poolSizes.data();
        descriptorPoolCI.maxSets = (2 + materialCount + meshCount) * renderUtils->UBCount;
        CHECK_RESULT(
                vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &descriptorPoolCI, nullptr, &resource.descriptorPool));

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
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                                     &resource.descriptorSetLayouts.scene));

            resource.descriptorSets.resize(renderUtils->UBCount);
            for (size_t i = 0; i < resource.descriptorSets.size(); i++) {
                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = resource.descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &resource.descriptorSetLayouts.scene;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                      &resource.descriptorSets[i]));

                std::array<VkWriteDescriptorSet, 5> writeDescriptorSets{};

                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].dstSet = resource.descriptorSets[i];
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].pBufferInfo = &resource.bufferScene.m_DescriptorBufferInfo;

                writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[1].descriptorCount = 1;
                writeDescriptorSets[1].dstSet = resource.descriptorSets[i];
                writeDescriptorSets[1].dstBinding = 1;
                writeDescriptorSets[1].pBufferInfo = &resource.bufferParams.m_DescriptorBufferInfo;

                writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[2].descriptorCount = 1;
                writeDescriptorSets[2].dstSet = resource.descriptorSets[i];
                writeDescriptorSets[2].dstBinding = 2;
                writeDescriptorSets[2].pImageInfo = &skyboxComponent.textures.irradianceCube.m_Descriptor;

                writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[3].descriptorCount = 1;
                writeDescriptorSets[3].dstSet = resource.descriptorSets[i];
                writeDescriptorSets[3].dstBinding = 3;
                writeDescriptorSets[3].pImageInfo = &skyboxComponent.textures.prefilteredCube.m_Descriptor;

                writeDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[4].descriptorCount = 1;
                writeDescriptorSets[4].dstSet = resource.descriptorSets[i];
                writeDescriptorSets[4].dstBinding = 4;
                writeDescriptorSets[4].pImageInfo = &skyboxComponent.textures.lutBrdf.m_Descriptor;

                vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, NULL);
            }
        }

        // Material (samplers)
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                                     &resource.descriptorSetLayouts.material));

            // Per-Material descriptor sets
            for (auto &material: component.model->materials) {
                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = resource.descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &resource.descriptorSetLayouts.material;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                      &material.descriptorSet));

                std::vector<VkDescriptorImageInfo> imageDescriptors = {
                        emptyTexture.m_Descriptor,
                        emptyTexture.m_Descriptor,
                        material.normalTexture ? material.normalTexture->m_Descriptor : emptyTexture.m_Descriptor,
                        material.occlusionTexture ? material.occlusionTexture->m_Descriptor : emptyTexture.m_Descriptor,
                        material.emissiveTexture ? material.emissiveTexture->m_Descriptor : emptyTexture.m_Descriptor
                };

                // TODO: glTF specs states that metallic roughness should be preferred, even if specular glosiness is present

                if (material.pbrWorkflows.metallicRoughness) {
                    if (material.baseColorTexture) {
                        imageDescriptors[0] = material.baseColorTexture->m_Descriptor;
                    }
                    if (material.metallicRoughnessTexture) {
                        imageDescriptors[1] = material.metallicRoughnessTexture->m_Descriptor;
                    }
                }

                if (material.pbrWorkflows.specularGlossiness) {
                    if (material.extension.diffuseTexture) {
                        imageDescriptors[0] = material.extension.diffuseTexture->m_Descriptor;
                    }
                    if (material.extension.specularGlossinessTexture) {
                        imageDescriptors[1] = material.extension.specularGlossinessTexture->m_Descriptor;
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

                vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, NULL);
            }

            // Model node (matrices)
            {
                std::vector<VkDescriptorSetLayoutBinding> nodeSetLayoutBindings = {
                        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                };
                VkDescriptorSetLayoutCreateInfo nodeSetLayoutCreateInfo{};
                nodeSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                nodeSetLayoutCreateInfo.pBindings = nodeSetLayoutBindings.data();
                nodeSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(nodeSetLayoutBindings.size());
                CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &nodeSetLayoutCreateInfo, nullptr,
                                                         &resource.descriptorSetLayouts.node));

                // Per-Node descriptor set
                for (auto &node: component.model->nodes) {
                    setupNodeDescriptorSet(node, resource.descriptorPool, &resource.descriptorSetLayouts.node);
                }
            }

            // Material Buffer
            {
                std::vector<VkDescriptorSetLayoutBinding> materialSetLayout = {
                        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                };
                VkDescriptorSetLayoutCreateInfo descriptorSetLayoutMaterialCreateInfo{};
                descriptorSetLayoutMaterialCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                descriptorSetLayoutMaterialCreateInfo.pBindings = materialSetLayout.data();
                descriptorSetLayoutMaterialCreateInfo.bindingCount = static_cast<uint32_t>(materialSetLayout.size());
                CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutMaterialCreateInfo, nullptr,
                                                         &resource.descriptorSetLayouts.materialBuffer));

                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = resource.descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &resource.descriptorSetLayouts.materialBuffer;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                      &resource.descriptorSetMaterials));

                VkWriteDescriptorSet writeDescriptorSet{};
                writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writeDescriptorSet.descriptorCount = 1;
                writeDescriptorSet.dstSet = resource.descriptorSetMaterials;
                writeDescriptorSet.dstBinding = 0;
                writeDescriptorSet.pBufferInfo = &resource.shaderMaterialBuffer.m_DescriptorBufferInfo;
                vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, 1, &writeDescriptorSet, 0, nullptr);
            }

        }
    }

    void DefaultPBRGraphicsPipelineComponent::setupNodeDescriptorSet(VkRender::Node *node, VkDescriptorPool pool, VkDescriptorSetLayout* layout) {
        if (node->mesh) {
            VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
            descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocInfo.descriptorPool = pool;
            descriptorSetAllocInfo.pSetLayouts = layout;
            descriptorSetAllocInfo.descriptorSetCount = 1;
            CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                  &node->mesh->uniformBuffer.descriptorSet));

            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.dstSet = node->mesh->uniformBuffer.descriptorSet;
            writeDescriptorSet.dstBinding = 0;
            writeDescriptorSet.pBufferInfo = &node->mesh->uniformBuffer.descriptor;

            vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, 1, &writeDescriptorSet, 0, nullptr);
        }
        for (auto &child: node->children) {
            setupNodeDescriptorSet(child, pool, layout);
        }
    }

    void DefaultPBRGraphicsPipelineComponent::setupPipelines(Resource& resource) {
        addPipelineSet(resource, "pbr", "pbr.vert.spv", "material_pbr.frag.spv");
        // KHR_materials_unlit
        addPipelineSet(resource, "unlit", "pbr.vert.spv", "material_unlit.frag.spv");
    }

    void DefaultPBRGraphicsPipelineComponent::addPipelineSet(Resource& resource, std::string prefix, std::string vertexShader,
                                                             std::string fragmentShader) {

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
        inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
        rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
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
        depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencilStateCI.front = depthStencilStateCI.back;
        depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS;

        VkPipelineViewportStateCreateInfo viewportStateCI{};
        viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateCI.viewportCount = 1;
        viewportStateCI.scissorCount = 1;

        VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
        multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleStateCI.rasterizationSamples = renderUtils->msaaSamples;


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
                resource.descriptorSetLayouts.scene, resource.descriptorSetLayouts.material, resource.descriptorSetLayouts.node,
                resource.descriptorSetLayouts.materialBuffer
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
                vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr, &resource.pipelineLayouts[prefix]));

        // Vertex bindings an attributes
        VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(VkRender::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT,       sizeof(float) * 6},
                {3, 0, VK_FORMAT_R32G32_SFLOAT,       sizeof(float) * 8},
                {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 10},
                {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 14},
                {6, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(float) * 18}
        };
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCI.vertexBindingDescriptionCount = 1;
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

        // Pipelines
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.layout = resource.pipelineLayouts[prefix];
        pipelineCI.renderPass = *renderUtils->renderPass;
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

        VkShaderModule vertModule{};
        VkShaderModule fragModule{};

        shaderStages[0] = Utils::loadShader(vulkanDevice->m_LogicalDevice, "spv/" + vertexShader,
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(vulkanDevice->m_LogicalDevice, "spv/" + fragmentShader,
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        VkPipeline pipeline{};

        // Default pipeline with back-face culling
        CHECK_RESULT(
                vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));
        resource.pipelines[prefix] = pipeline;
        // Double sided
        rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
        CHECK_RESULT(
                vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));
        resource.pipelines[prefix + "_double_sided"] = pipeline;
        // Alpha blending
        rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
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
        CHECK_RESULT(
                vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr, &pipeline));
        resource.pipelines[prefix + "_alpha_blending"] = pipeline;

        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
        }
    }


    void DefaultPBRGraphicsPipelineComponent::createMaterialBuffer(Resource& resource, const VkRender::GLTFModelComponent &component) {
        std::vector<ShaderMaterial> shaderMaterials{};
        for (auto &material: component.model->materials) {
            ShaderMaterial shaderMaterial{};

            shaderMaterial.emissiveFactor = material.emissiveFactor;
            // To save space, availabilty and texture coordinate set are combined
            // -1 = texture not used for this material, >= 0 texture used and index of texture coordinate set
            shaderMaterial.colorTextureSet =
                    material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;
            shaderMaterial.normalTextureSet = material.normalTexture != nullptr ? material.texCoordSets.normal : -1;
            shaderMaterial.occlusionTextureSet =
                    material.occlusionTexture != nullptr ? material.texCoordSets.occlusion : -1;
            shaderMaterial.emissiveTextureSet =
                    material.emissiveTexture != nullptr ? material.texCoordSets.emissive : -1;
            shaderMaterial.alphaMask = static_cast<float>(material.alphaMode == VkRender::Material::ALPHAMODE_MASK);
            shaderMaterial.alphaMaskCutoff = material.alphaCutoff;
            shaderMaterial.emissiveStrength = material.emissiveStrength;

            // TODO: glTF specs states that metallic roughness should be preferred, even if specular glosiness is present

            if (material.pbrWorkflows.metallicRoughness) {
                // Metallic roughness workflow
                shaderMaterial.workflow = static_cast<float>(PBR_WORKFLOW_METALLIC_ROUGHNESS);
                shaderMaterial.baseColorFactor = material.baseColorFactor;
                shaderMaterial.metallicFactor = material.metallicFactor;
                shaderMaterial.roughnessFactor = material.roughnessFactor;
                shaderMaterial.PhysicalDescriptorTextureSet =
                        material.metallicRoughnessTexture != nullptr ? material.texCoordSets.metallicRoughness : -1;
                shaderMaterial.colorTextureSet =
                        material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;
            }

            if (material.pbrWorkflows.specularGlossiness) {
                // Specular glossiness workflow
                shaderMaterial.workflow = static_cast<float>(PBR_WORKFLOW_SPECULAR_GLOSINESS);
                shaderMaterial.PhysicalDescriptorTextureSet = material.extension.specularGlossinessTexture != nullptr
                                                              ? material.texCoordSets.specularGlossiness : -1;
                shaderMaterial.colorTextureSet =
                        material.extension.diffuseTexture != nullptr ? material.texCoordSets.baseColor : -1;
                shaderMaterial.diffuseFactor = material.extension.diffuseFactor;
                shaderMaterial.specularFactor = glm::vec4(material.extension.specularFactor, 1.0f);
            }

            shaderMaterials.push_back(shaderMaterial);
        }

        if (resource.shaderMaterialBuffer.m_Buffer != VK_NULL_HANDLE) {
            resource.shaderMaterialBuffer.destroy();
        }
        VkDeviceSize bufferSize = shaderMaterials.size() * sizeof(ShaderMaterial);
        Buffer stagingBuffer;
        CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                bufferSize, &stagingBuffer.m_Buffer, &stagingBuffer.m_Memory,
                                                shaderMaterials.data()));

        CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bufferSize,
                                                &resource.shaderMaterialBuffer.m_Buffer, &resource.shaderMaterialBuffer.m_Memory));

        // Copy from staging buffers
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion{};
        copyRegion.size = bufferSize;
        vkCmdCopyBuffer(copyCmd, stagingBuffer.m_Buffer, resource.shaderMaterialBuffer.m_Buffer, 1, &copyRegion);
        vulkanDevice->flushCommandBuffer(copyCmd, vulkanDevice->m_TransferQueue, true);
        stagingBuffer.m_Device = vulkanDevice->m_LogicalDevice;

        // Update descriptor
        resource.shaderMaterialBuffer.m_DescriptorBufferInfo.buffer = resource.shaderMaterialBuffer.m_Buffer;
        resource.shaderMaterialBuffer.m_DescriptorBufferInfo.offset = 0;
        resource.shaderMaterialBuffer.m_DescriptorBufferInfo.range = bufferSize;
        resource.shaderMaterialBuffer.m_Device = vulkanDevice->m_LogicalDevice;
    }
};