//
// Created by magnus on 4/24/24.
//

#ifndef MULTISENSE_VIEWER_CAMERAGRAPHICSPIPELINECOMPONENT_H
#define MULTISENSE_VIEWER_CAMERAGRAPHICSPIPELINECOMPONENT_H

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {

    struct CameraGraphicsPipelineComponent {
        CameraGraphicsPipelineComponent() = default;

        CameraGraphicsPipelineComponent(const CameraGraphicsPipelineComponent &) = default;

        CameraGraphicsPipelineComponent &
        operator=(const CameraGraphicsPipelineComponent &other) { return *this; }

        explicit CameraGraphicsPipelineComponent(VkRender::RenderUtils *utils) {
            renderUtils = utils;
            vulkanDevice = utils->device;

            uint32_t numSwapChainImages = renderUtils->UBCount;
            uint32_t numResourcesToAllocate = 1;
            std::vector<VkRenderPass> passes = {*renderUtils->renderPass};

            resources.resize(numResourcesToAllocate);
            renderData.resize(numSwapChainImages);
            setupUniformBuffers();
            setupDescriptors();

            for (size_t i = 0; i < resources.size(); ++i) {
                resources[i].res.resize(numSwapChainImages);
                resources[i].renderPass = passes[i];
                resources[i].vertexShader = "CameraGizmo.vert.spv";
                resources[i].fragmentShader = "CameraGizmo.frag.spv";
                for (size_t j = 0; j < resources[i].res.size(); ++j) {
                    setupPipeline(resources[i].res[j], resources[i],
                                  renderData[j].descriptorSetLayout);
                }
            }
        }


        ~CameraGraphicsPipelineComponent() {


            for (auto &resource: resources) {
                for (auto &res: resource.res) {
                    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, res.pipeline, nullptr);
                    vkDestroyPipeline(vulkanDevice->m_LogicalDevice, res.pipelineFill, nullptr);
                    vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, res.pipelineLayout, nullptr);
                }
            }

            for (auto &res: renderData) {
                vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, res.descriptorSetLayout, nullptr);
                vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, res.descriptorPool, nullptr);
            }

        };


        bool markedForDeletion = false;
        struct Resource {

            VkPipeline pipeline{};
            VkPipeline pipelineFill{};
            VkPipelineLayout pipelineLayout{};

            bool busy = false;
            bool requestIdle = false;
        };

        struct RenderResource {
            std::vector<Resource> res{};
            VkRenderPass renderPass = VK_NULL_HANDLE;
            std::string vertexShader;
            std::string fragmentShader;
        };

        struct RenderData {
            Buffer uboVertices;
            Buffer ubo;
            VkDescriptorPool descriptorPool{};
            std::vector<VkDescriptorSet> descriptorSets;
            VkDescriptorSetLayout descriptorSetLayout{};
            VkRender::UBOMatrix mvp;
            VkRender::UBOCamera vertices;
        };

        std::vector<RenderResource> resources;
        std::vector<RenderData> renderData;


        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex, uint32_t renderPassIndex = 0) {
            if (markedForDeletion || resources[renderPassIndex].res[cbIndex].requestIdle) {
                resources[renderPassIndex].res[cbIndex].busy = false;
                return;
            }

            vkCmdBindPipeline(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              resources[renderPassIndex].res[cbIndex].pipeline);

            // TODO Make dynamic with amount of renderpassess allocated
            vkCmdBindDescriptorSets(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    resources[renderPassIndex].res[cbIndex].pipelineLayout, 0, static_cast<uint32_t>(1),
                                    &renderData[cbIndex].descriptorSets[cbIndex], 0, nullptr);

            vkCmdDraw(commandBuffer->buffers[cbIndex], 18, 1, 0, 0);

            vkCmdBindPipeline(commandBuffer->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              resources[renderPassIndex].res[cbIndex].pipelineFill);

            vkCmdDraw(commandBuffer->buffers[cbIndex], 3, 1, 18, 0);


            resources[renderPassIndex].res[cbIndex].busy = true;
        }

        void update() {
            for (auto &resource: resources) {
                for (size_t i = 0; i < resource.res.size(); ++i) {
                    if (!resource.res[i].busy && resource.res[i].requestIdle) {
                        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, resource.res[i].pipeline, nullptr);
                        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, resource.res[i].pipelineFill, nullptr);
                        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice,
                                                resource.res[i].pipelineLayout, nullptr);

                        setupPipeline(resource.res[i], resource,
                                      renderData[i].descriptorSetLayout);
                    }
                }
            }

            memcpy(renderData[renderUtils->swapchainIndex].ubo.mapped,
                   &renderData[renderUtils->swapchainIndex].mvp, sizeof(VkRender::UBOMatrix));
            memcpy(renderData[renderUtils->swapchainIndex].uboVertices.mapped,
                   &renderData[renderUtils->swapchainIndex].vertices, sizeof(VkRender::UBOCamera));
        }

        void updateGraphicsPipeline() {
            for (auto &resource: resources) {
                for (auto &res: resource.res) {
                    res.requestIdle = true;
                }
            }
        }

    private:
        void setupUniformBuffers() {
            for (auto &resource: renderData) {
                vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                           &resource.uboVertices, sizeof(VkRender::UBOCamera));
                vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                           &resource.ubo, sizeof(VkRender::UBOMatrix));

                resource.uboVertices.map();
                resource.ubo.map();

                resource.vertices.positions = {
                        // Base (CCW from top)
                        glm::vec4(-0.5, 0.5, 0, 1.0), // D
                        glm::vec4(-0.5, -0.5, 0, 1.0), // A
                        glm::vec4(0.5, -0.5, 0, 1.0), // B
                        glm::vec4(-0.5, 0.5, 0, 1.0), // D
                        glm::vec4(0.5, -0.5, 0, 1.0), // B
                        glm::vec4(0.5, 0.5, 0, 1.0), // C

                        // Side 1
                        glm::vec4(-0.5, -0.5, 0, 1.0), // A
                        glm::vec4(0, 0, 0.866, 1.0), // E
                        glm::vec4(0.5f, -0.5f, 0, 1.0f), // B

                        // Side 2
                        glm::vec4(0.5, -0.5, 0, 1.0), // B
                        glm::vec4(0, 0, 0.866, 1.0), // E
                        glm::vec4(0.5, 0.5, 0, 1.0), // C

                        // Side 3
                        glm::vec4(0.5, 0.5, 0, 1.0), // C
                        glm::vec4(0, 0, 0.866, 1.0), // E
                        glm::vec4(-0.5, 0.5, 0, 1.0), // D

                        // Side 4
                        glm::vec4(-0.5, 0.5, 0, 1.0), // D
                        glm::vec4(0, 0, 0.866, 1.0), // E
                        glm::vec4(-0.5, -0.5, 0, 1.0), // A

                        // Top indicator
                        glm::vec4(-0.4, -0.6, 0, 1.0), // D
                        glm::vec4(0.4, -0.6, 0, 1.0), // E
                        glm::vec4(0, -1.0, 0, 1.0) // A
                };

            }
        }

        void setupDescriptors() {
            for (auto &resource: renderData) {

                std::vector<VkDescriptorPoolSize> poolSizes = {
                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, renderUtils->UBCount},
                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, renderUtils->UBCount}
                };
                VkDescriptorPoolCreateInfo descriptorPoolCI{};
                descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
                descriptorPoolCI.pPoolSizes = poolSizes.data();
                descriptorPoolCI.maxSets = renderUtils->UBCount;
                CHECK_RESULT(
                        vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &descriptorPoolCI, nullptr,
                                               &resource.descriptorPool));

                // Scene (matrices and environment maps)
                {
                    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT |
                                                                      VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    };
                    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
                    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                    descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
                    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
                    CHECK_RESULT(
                            vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
                                                        &resource.descriptorSetLayout));

                    resource.descriptorSets.resize(renderUtils->UBCount);
                    for (size_t i = 0; i < resource.descriptorSets.size(); i++) {
                        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                        descriptorSetAllocInfo.descriptorPool = resource.descriptorPool;
                        descriptorSetAllocInfo.pSetLayouts = &resource.descriptorSetLayout;
                        descriptorSetAllocInfo.descriptorSetCount = 1;
                        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                                              &resource.descriptorSets[i]));

                        std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};

                        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                        writeDescriptorSets[0].descriptorCount = 1;
                        writeDescriptorSets[0].dstSet = resource.descriptorSets[i];
                        writeDescriptorSets[0].dstBinding = 0;
                        writeDescriptorSets[0].pBufferInfo = &resource.ubo.m_DescriptorBufferInfo;

                        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                        writeDescriptorSets[1].descriptorCount = 1;
                        writeDescriptorSets[1].dstSet = resource.descriptorSets[i];
                        writeDescriptorSets[1].dstBinding = 1;
                        writeDescriptorSets[1].pBufferInfo = &resource.uboVertices.m_DescriptorBufferInfo;

                        vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice,
                                               static_cast<uint32_t>(writeDescriptorSets.size()),
                                               writeDescriptorSets.data(), 0, nullptr);
                    }
                }
            }
        }

        VkRender::RenderUtils *renderUtils;
        VulkanDevice *vulkanDevice;

        void setupPipeline(Resource &resource, RenderResource &rr, VkDescriptorSetLayout &pT) {
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
            inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo rasterStateCI{};
            rasterStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterStateCI.polygonMode = VK_POLYGON_MODE_LINE;
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
                    pT
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
                    vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutCI, nullptr,
                                           &resource.pipelineLayout));

            // Vertex bindings an attributes

            std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            };
            VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
            vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputStateCI.vertexBindingDescriptionCount = 0;
            vertexInputStateCI.vertexAttributeDescriptionCount = 0;

            // Pipelines
            std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

            VkGraphicsPipelineCreateInfo pipelineCI{};
            pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCI.layout = resource.pipelineLayout;
            pipelineCI.renderPass = rr.renderPass;
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


            shaderStages[0] = Utils::loadShader(vulkanDevice->m_LogicalDevice, "spv/" + rr.vertexShader,
                                                VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
            shaderStages[1] = Utils::loadShader(vulkanDevice->m_LogicalDevice, "spv/" + rr.fragmentShader,
                                                VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

            // Default pipeline with back-face culling
            CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                                   &resource.pipeline));

            // Default pipeline with back-face culling
            rasterStateCI.polygonMode = VK_POLYGON_MODE_FILL;
            CHECK_RESULT(vkCreateGraphicsPipelines(vulkanDevice->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                                   &resource.pipelineFill));

            for (auto shaderStage: shaderStages) {
                vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, shaderStage.module, nullptr);
            }
            // If the pipeline was updated and we had previously requested it to be idle
            resource.requestIdle = false;
        }
    };


};

#endif //MULTISENSE_VIEWER_CAMERAGRAPHICSPIPELINECOMPONENT_H
