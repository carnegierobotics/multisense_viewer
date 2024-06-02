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
#include "Viewer/Renderer/Components/RenderComponents/RenderBase.h"
#include "Viewer/Renderer/Components.h"

namespace VkRender {

    struct CameraGraphicsPipelineComponent : RenderBase {
        CameraGraphicsPipelineComponent() = default;

        CameraGraphicsPipelineComponent(const CameraGraphicsPipelineComponent &) = delete;

        CameraGraphicsPipelineComponent& operator=(const CameraGraphicsPipelineComponent&) = delete;

        explicit CameraGraphicsPipelineComponent(VkRender::RenderUtils *utils) {
            m_numSwapChainImages = utils->swapchainImages;
            m_vulkanDevice = utils->device;
            // Number of resources per render pass
            m_renderData.resize(m_numSwapChainImages);

            // Assume we get a modelComponent that has vertex and index buffers in gpu memory. We need to create graphics resources which are:
            // Descriptor sets: pool, layout, sets
            // Uniform Buffers:
            setupUniformBuffers();
            setupDescriptors();
            // First create normal render pass resources
            // Graphics pipelines
            for (auto &data: m_renderData) {

                setupPipeline(data, RENDER_PASS_COLOR, "CameraGizmo.vert.spv", "CameraGizmo.frag.spv", utils->msaaSamples,
                              *utils->renderPass);

                setupPipeline(data, RENDER_PASS_SECOND, "CameraGizmo.vert.spv", "CameraGizmo.frag.spv", utils->msaaSamples,
                              *utils->renderPass); // TODO set correct render pass
            }
        }
        VkRender::UBOMatrix mvp{};
        VkRender::UBOCamera vertices{};

        std::vector<DefaultRenderData> m_renderData;
        VulkanDevice *m_vulkanDevice = nullptr;
        uint32_t m_numSwapChainImages = 0;
        bool resourcesDeleted = false;

        void updateTransform(const TransformComponent& transform){
            mvp.model = transform.GetTransform();
        }
        void updateView(const Camera& camera){
            mvp.view = camera.matrices.view;
            mvp.projection = camera.matrices.perspective;
            mvp.camPos = camera.pose.pos;

        }

        ~CameraGraphicsPipelineComponent() {
            if (!resourcesDeleted)
                cleanUp(0, true);
        };

        void draw(CommandBuffer *cmdBuffers) override {
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


            vkCmdDraw(cmdBuffers->buffers[cbIndex], 21, 1, 0, 0);

            /*
            vkCmdBindPipeline(cmdBuffers->buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                              m_renderData[cbIndex].pipeline[renderPassType]);

            vkCmdDraw(cmdBuffers->buffers[cbIndex], 3, 1, 18, 0);
            */


            m_renderData[cbIndex].busy[renderPassType] = true;
        }

        bool cleanUp(uint32_t currentFrame, bool force=false) override {

            pauseRendering();
            bool cleanUp = true;

            for (auto &data: m_renderData) {
                for (const auto &busy: data.busy) {
                    if (busy.second) {
                        cleanUp = false;
                    }
                }
            }
            if (cleanUp || force)  {
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
                resourcesDeleted = true;
            } else {
                Log::Logger::getInstance()->trace("Waiting to clean up vulkan resources for DefaultGraphicsPipeline");
                for (auto &busy: m_renderData[currentFrame].busy) {
                    busy.second = false;
                }
            }

            return cleanUp;
        }

        void pauseRendering() override {

        }

        void update(uint32_t currentFrame) override {
            memcpy(m_renderData[currentFrame].mvpBuffer.mapped,
                   &mvp, sizeof(VkRender::UBOMatrix));
            memcpy(m_renderData[currentFrame].fragShaderParamsBuffer.mapped,
                   &vertices, sizeof(VkRender::UBOCamera));

        }

        void update() {


        }

    private:
        void setupUniformBuffers() {
            for (auto &resource: m_renderData) {
                m_vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                             &resource.fragShaderParamsBuffer, sizeof(VkRender::UBOCamera));
                m_vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                             &resource.mvpBuffer, sizeof(VkRender::UBOMatrix));

                resource.mvpBuffer.map();
                resource.fragShaderParamsBuffer.map();
                float a = 0.5;
                float h = 2.0;
                vertices.positions = {
                        // Base (CCW from top)
                        glm::vec4(-a, a, 0, 1.0), // D
                        glm::vec4(-a, -a, 0, 1.0), // A
                        glm::vec4(a, -a, 0, 1.0), // B
                        glm::vec4(-a, a, 0, 1.0), // D
                        glm::vec4(a, -a, 0, 1.0), // B
                        glm::vec4(a, a, 0, 1.0), // C

                        // Side 1
                        glm::vec4(-a, -a, 0, 1.0), // A
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(a, -a, 0, 1.0f), // B

                        // Side 2
                        glm::vec4(a, -a, 0, 1.0), // B
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(a, a, 0, 1.0), // C

                        // Side 3
                        glm::vec4(a, a, 0, 1.0), // C
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(-a, a, 0, 1.0), // D

                        // Side 4
                        glm::vec4(-a, a, 0, 1.0), // D
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(-a, -a, 0, 1.0), // A

                        // Top indicator
                        glm::vec4(-0.4, 0.6, 0, 1.0), // D
                        glm::vec4(0.4, 0.6, 0, 1.0), // E
                        glm::vec4(0, 1.0, 0, 1.0) // A
                };

            }
        }

        void setupDescriptors() {
            for (auto &resource: m_renderData) {

                std::vector<VkDescriptorPoolSize> poolSizes = {
                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m_numSwapChainImages},
                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m_numSwapChainImages}
                };
                VkDescriptorPoolCreateInfo descriptorPoolCI{};
                descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
                descriptorPoolCI.pPoolSizes = poolSizes.data();
                descriptorPoolCI.maxSets = m_numSwapChainImages;
                CHECK_RESULT(
                        vkCreateDescriptorPool(m_vulkanDevice->m_LogicalDevice, &descriptorPoolCI, nullptr,
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
                            vkCreateDescriptorSetLayout(m_vulkanDevice->m_LogicalDevice, &descriptorSetLayoutCI, nullptr,
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

                        std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};

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

                        vkUpdateDescriptorSets(m_vulkanDevice->m_LogicalDevice,
                                               static_cast<uint32_t>(writeDescriptorSets.size()),
                                               writeDescriptorSets.data(), 0, nullptr);
                    }
                }
            }
        }



        void setupPipeline(DefaultRenderData &data, RenderPassType type,
                                                              const std::string &vertexShader,
                                                              const std::string &fragmentShader,
                                                              VkSampleCountFlagBits sampleCountFlagBits,
                                                              VkRenderPass renderPass) {
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


            VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
            vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputStateCI.vertexBindingDescriptionCount = 0;
            vertexInputStateCI.vertexAttributeDescriptionCount = 0;

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
    };


};

#endif //MULTISENSE_VIEWER_CAMERAGRAPHICSPIPELINECOMPONENT_H
