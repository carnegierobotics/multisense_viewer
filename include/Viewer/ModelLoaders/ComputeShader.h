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

class ComputeShader{
public:
    ComputeShader(){

    }

    VulkanDevice *vulkanDevice{};

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
        buffer.resize(numBuffers);
        memory.resize(numBuffers);
        // Create staging buffers
        // Vertex m_DataPtr
        CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                bufferSize,
                &stagingBuffer,
                &stagingMemory,
                reinterpret_cast<const void *>(particles.data())))

        for (size_t i = 0; i < numBuffers; i++) {

            CHECK_RESULT(vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    bufferSize,
                    &buffer[i],
                    &memory[i] ));
            // Copy from staging buffers
            VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkBufferCopy copyRegion = {};
            copyRegion.size = bufferSize;
            vkCmdCopyBuffer(copyCmd, stagingBuffer, buffer[i], 1, &copyRegion);
            vulkanDevice->flushCommandBuffer(copyCmd, vulkanDevice->m_TransferQueue, true);
        }
        vkDestroyBuffer(vulkanDevice->m_LogicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(vulkanDevice->m_LogicalDevice, stagingMemory, nullptr);
    }


    void createDescriptorSetLayout(){
        std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};
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

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(vulkanDevice->m_LogicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createDescriptorSetPool(uint32_t count){
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         count},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         2 * count},

        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(
                vkCreateDescriptorPool(vulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));
    }

    void createDescriptorSets(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo){
        descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              descriptors.data()));

        for (size_t i = 0; i < count; i++) {

            std::vector<VkWriteDescriptorSet> descriptorWrites(3);

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptors[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &ubo[i].bufferTwo.m_DescriptorBufferInfo;

            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            //storageBufferInfoLastFrame.buffer = buffer[];
            storageBufferInfoLastFrame.buffer = buffer[i];
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
            storageBufferInfoCurrentFrame.buffer = buffer[(i + 1) % count];
            storageBufferInfoCurrentFrame.offset = 0;
            storageBufferInfoCurrentFrame.range = sizeof(VkRender::Particle) * PARTICLE_COUNT;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptors[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;

            vkUpdateDescriptorSets(vulkanDevice->m_LogicalDevice, 3, descriptorWrites.data(), 0, nullptr);
        }
    }

    void createComputePipeline(VkPipelineShaderStageCreateInfo computeShaderStageInfo){
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(vulkanDevice->m_LogicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(vulkanDevice->m_LogicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
    }


    void recordDrawCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame){
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording compute command buffer!");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptors[currentFrame], 0, nullptr);

        //Log::Logger::getInstance()->trace("Binding descriptors: {} in compute-call", currentFrame);

        vkCmdDispatch(commandBuffer, PARTICLE_COUNT / 256, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record compute command buffer!");
        }

    }

    ~ComputeShader(){

        for(auto& b : buffer){
            vkDestroyBuffer(vulkanDevice->m_LogicalDevice, b, nullptr);
        }
        for(auto& m : memory){
            vkFreeMemory(vulkanDevice->m_LogicalDevice, m, nullptr);
        }

        vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
        vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);
        vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice,descriptorSetLayout ,nullptr);
    }

    std::vector<VkBuffer> buffer{};
    std::vector<VkDeviceMemory> memory{};

    private:
        int PARTICLE_COUNT = 4096;

    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorPool descriptorPool{};
    std::vector<VkDescriptorSet> descriptors{};

    VkPipeline pipeline{};
    VkPipelineLayout pipelineLayout{};
};

#endif //MULTISENSE_VIEWER_COMPUTESHADER_H
