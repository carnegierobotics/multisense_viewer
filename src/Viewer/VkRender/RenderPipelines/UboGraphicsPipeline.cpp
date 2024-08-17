//
// Created by mgjer on 14/08/2024.
//


#include "Viewer/VkRender/RenderPipelines/UboGraphicsPipeline.h"
#include "Viewer/VkRender/Renderer.h"

namespace VkRender {

    UboGraphicsPipeline::UboGraphicsPipeline(Renderer &m_context,
                                             const RenderPassInfo &renderPassInfo)
            : m_vulkanDevice(m_context.vkDevice()),
              m_renderPassInfo(std::move(renderPassInfo)) {

        m_numSwapChainImages = m_context.swapChainBuffers().size();
        m_vulkanDevice = m_context.vkDevice();
        m_vertexShader = "CameraGizmo.vert.spv";
        m_fragmentShader = "CameraGizmo.frag.spv";

        m_renderData.resize(m_numSwapChainImages);

        setupUniformBuffers();
        setupDescriptors();
        setupPipeline();
    }

    UboGraphicsPipeline::~UboGraphicsPipeline() {
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        VkFence fence;
        VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
        vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence);

        VkDescriptorSetLayout layout =  m_sharedRenderData.descriptorSetLayout;
        VkDescriptorPool pool =  m_sharedRenderData.descriptorPool;

        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, layout, pool]() {
                    vkDestroyDescriptorSetLayout(logicalDevice, layout, nullptr);
                    vkDestroyDescriptorPool(logicalDevice, pool, nullptr);

                },
                fence);

    }

    void UboGraphicsPipeline::update(uint32_t currentFrame) {
        memcpy(m_renderData[currentFrame].mvpBuffer.mapped,
               &m_vertexParams, sizeof(VkRender::UBOMatrix));



    }
    void UboGraphicsPipeline::updateTransform(const TransformComponent &transform) {
        m_vertexParams.model = transform.GetTransform();

    }
    void UboGraphicsPipeline::updateView(const Camera &camera) {
        m_vertexParams.view = camera.matrices.view;
        m_vertexParams.projection = camera.matrices.perspective;
        m_vertexParams.camPos = camera.pose.pos;
    }


    void UboGraphicsPipeline::draw(CommandBuffer &cmdBuffers) {
        const uint32_t &cbIndex = *cmdBuffers.frameIndex
                ;
        vkCmdBindPipeline(cmdBuffers.buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, m_sharedRenderData.graphicsPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmdBuffers.buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_sharedRenderData.graphicsPipeline->getPipelineLayout(), 0, static_cast<uint32_t>(1),
                                &m_renderData[cbIndex].descriptorSet, 0, nullptr);
        vkCmdDraw(cmdBuffers.buffers[cbIndex], m_vertexCount, 1, 0, 0);
    }

    void UboGraphicsPipeline::bind(MeshComponent &modelComponent) {
        for (auto& renderData : m_renderData){
            memcpy(renderData.fragShaderParamsBuffer.mapped,
                   modelComponent.getCameraModelMesh().positions.data(), sizeof(VkRender::UBOCamera));

        }
        m_vertexCount = modelComponent.getCameraModelMesh().positions.size();
    }

    void UboGraphicsPipeline::setupUniformBuffers() {
        for (auto &data: m_renderData) {
            m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        &data.mvpBuffer, sizeof(VkRender::UBOMatrix));

            data.mvpBuffer.map();
            m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        &data.fragShaderParamsBuffer, sizeof(VkRender::UBOCamera));

            data.fragShaderParamsBuffer.map();
        }
    }
    void UboGraphicsPipeline::setupDescriptors() {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         m_numSwapChainImages * 2},
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
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
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
                std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};
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
                vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice,
                                       static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, nullptr);
            }
        }
    }

    void UboGraphicsPipeline::setupPipeline() {
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCI.vertexBindingDescriptionCount = 0;
        vertexInputStateCI.vertexAttributeDescriptionCount = 0;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages(2);
        VkShaderModule vertModule{};
        VkShaderModule fragModule{};
        shaderStages[0] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_vertexShader,
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_fragmentShader,
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);
        VulkanGraphicsPipelineCreateInfo createInfo( m_renderPassInfo.renderPass, m_vulkanDevice);
        createInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_LINE, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        createInfo.msaaSamples = m_renderPassInfo.sampleCount;
        createInfo.shaders = shaderStages;
        createInfo.descriptorSetLayout = m_sharedRenderData.descriptorSetLayout;
        createInfo.vertexInputState = vertexInputStateCI;
        m_sharedRenderData.graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(createInfo);
        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(m_vulkanDevice.m_LogicalDevice, shaderStage.module, nullptr);
        }

    }

}