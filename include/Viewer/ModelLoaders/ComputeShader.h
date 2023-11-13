//
// Created by magnus on 11/7/23.
//

#ifndef MULTISENSE_VIEWER_COMPUTESHADER_H
#define MULTISENSE_VIEWER_COMPUTESHADER_H

#include <random>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>
#include <glm/geometric.hpp>
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Core/Definitions.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Scripts/Private/TextureDataDef.h"

class ComputeShader {
public:
    ComputeShader() {

    }

    VulkanDevice *m_VulkanDevice{};

    void createBuffers(uint32_t numBuffers) {

        // Initialize particles
        std::default_random_engine rndEngine((unsigned) time(nullptr));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

        // Initial particle positions on a circle
        std::vector<VkRender::Particle> particles(PARTICLE_COUNT);
        float HEIGHT = 720.0f;
        float WIDTH = 1280.0f;
        for (auto &particle: particles) {
            float r = 0.25f * sqrtf(rndDist(rndEngine));
            float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
            float x = r * cosf(theta) * HEIGHT / WIDTH;
            float y = r * sinf(theta);
            particle.position = glm::vec2(x, y);
            particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);

        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;

        VkDeviceSize bufferSize = sizeof(VkRender::Particle) * PARTICLE_COUNT;
        m_Buffer.resize(numBuffers);
        m_Memory.resize(numBuffers);
        // Create staging buffers
        // Vertex m_DataPtr
        CHECK_RESULT(m_VulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                bufferSize,
                &stagingBuffer,
                &stagingMemory,
                reinterpret_cast<const void *>(particles.data())))

        for (size_t i = 0; i < numBuffers; i++) {

            CHECK_RESULT(m_VulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    bufferSize,
                    &m_Buffer[i],
                    &m_Memory[i]));
            // Copy from staging buffers
            VkCommandBuffer copyCmd = m_VulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkBufferCopy copyRegion = {};
            copyRegion.size = bufferSize;
            vkCmdCopyBuffer(copyCmd, stagingBuffer, m_Buffer[i], 1, &copyRegion);
            m_VulkanDevice->flushCommandBuffer(copyCmd, m_VulkanDevice->m_TransferQueue, true);
        }
        vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(m_VulkanDevice->m_LogicalDevice, stagingMemory, nullptr);
    }


    void createDescriptorSetLayout() {
        std::array<VkDescriptorSetLayoutBinding, 6> layoutBindings{};
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[4].pImmutableSamplers = nullptr;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        layoutBindings[5].binding = 5;
        layoutBindings[5].descriptorCount = 1;
        layoutBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[5].pImmutableSamplers = nullptr;
        layoutBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(m_VulkanDevice->m_LogicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createDescriptorSetPool(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},

        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(
                vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));
    }

    void createDescriptorSets(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              descriptors.data()));

        for (size_t i = 0; i < count; i++) {

            std::vector<VkWriteDescriptorSet> descriptorWrites(6);

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptors[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &ubo[i].bufferTwo.m_DescriptorBufferInfo;

            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            //storageBufferInfoLastFrame.buffer = buffer[];
            storageBufferInfoLastFrame.buffer = m_Buffer[i];
            storageBufferInfoLastFrame.offset = 0;
            storageBufferInfoLastFrame.range = sizeof(VkRender::Particle) * PARTICLE_COUNT;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptors[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

            VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
            storageBufferInfoCurrentFrame.buffer = m_Buffer[(i + 1) % count];
            storageBufferInfoCurrentFrame.offset = 0;
            storageBufferInfoCurrentFrame.range = sizeof(VkRender::Particle) * PARTICLE_COUNT;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptors[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = descriptors[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pImageInfo = &m_TextureComputeLeftInput[i]->m_Descriptor;

            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = descriptors[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pImageInfo = &m_TextureComputeRightInput[i]->m_Descriptor;

            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = descriptors[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pImageInfo = &m_TextureComputeTargets[i].m_Descriptor;

            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, 6, descriptorWrites.data(), 0, nullptr);
        }
    }

    void createTextureTarget(uint32_t width, uint32_t height, uint32_t framesInFlight, uint32_t numTextures = 6) {
        m_TextureComputeTargets.resize(framesInFlight * numTextures);
        m_TextureComputeLeftInput.resize(framesInFlight * numTextures);
        m_TextureComputeRightInput.resize(framesInFlight * numTextures);
        uint8_t *array = new uint8_t[width * height](); // The parentheses initialize all elements to zero.
        for (uint32_t i = 0; i < framesInFlight; ++i) {
            m_TextureComputeLeftInput[i] = std::make_unique<TextureVideo>(width, height, m_VulkanDevice,
                                                                                       VK_IMAGE_LAYOUT_GENERAL,
                                                                                       VK_FORMAT_R8_UNORM,
                                                                                       VK_IMAGE_USAGE_SAMPLED_BIT |
                                                                                       VK_IMAGE_USAGE_STORAGE_BIT,
                                                                                       true);
            m_TextureComputeRightInput[i] = std::make_unique<TextureVideo>(width, height, m_VulkanDevice,
                                                                                        VK_IMAGE_LAYOUT_GENERAL,
                                                                                        VK_FORMAT_R8_UNORM,
                                                                                        VK_IMAGE_USAGE_SAMPLED_BIT |
                                                                                        VK_IMAGE_USAGE_STORAGE_BIT,
                                                                                        true);
        }

        for (uint32_t i = 0; i < framesInFlight * numTextures; ++i) {
            m_TextureComputeTargets[i].fromBuffer(array, width * height, VK_FORMAT_R8_UNORM, width, height,
                                                  m_VulkanDevice,
                                                  m_VulkanDevice->m_TransferQueue, VK_FILTER_LINEAR,
                                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                  VK_IMAGE_LAYOUT_GENERAL,
                                                  true);
        }
        delete[] array;
    }

    void createComputePipeline(VkPipelineShaderStageCreateInfo computeShaderStageInfo) {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(m_VulkanDevice->m_LogicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute m_Pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(m_VulkanDevice->m_LogicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                     &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute m_Pipeline!");
        }
    }


    void recordDrawCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording compute command buffer!");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                &descriptors[currentFrame], 0, nullptr);

        //Log::Logger::getInstance()->trace("Binding descriptors: {} in compute-call", currentFrame);

        vkCmdDispatch(commandBuffer, m_TextureComputeLeftInput[currentFrame]->m_Width / 16,  m_TextureComputeLeftInput[currentFrame]->m_Height / 16, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record compute command buffer!");
        }

    }

    ~ComputeShader() {

        for (auto &b: m_Buffer) {
            vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, b, nullptr);
        }
        for (auto &m: m_Memory) {
            vkFreeMemory(m_VulkanDevice->m_LogicalDevice, m, nullptr);
        }

        vkDestroyPipelineLayout(m_VulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
        vkDestroyPipeline(m_VulkanDevice->m_LogicalDevice, pipeline, nullptr);
        vkDestroyDescriptorPool(m_VulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(m_VulkanDevice->m_LogicalDevice, descriptorSetLayout, nullptr);
    }

    std::vector<VkBuffer> m_Buffer{};
    std::vector<Texture2D> m_TextureComputeTargets{};
    std::vector<std::unique_ptr<TextureVideo>> m_TextureComputeLeftInput{};
    std::vector<std::unique_ptr<TextureVideo>> m_TextureComputeRightInput{};
    std::vector<VkDeviceMemory> m_Memory{};



private:
    int PARTICLE_COUNT = 4096;

    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    std::vector<VkDescriptorSet> descriptors{};

    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};
};


#endif //MULTISENSE_VIEWER_COMPUTESHADER_H
