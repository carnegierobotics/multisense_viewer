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
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Scripts/Private/TextureDataDef.h"

class ComputeShader {
public:
    ComputeShader() = default;

    VulkanDevice *m_VulkanDevice{};

    void createBuffers(uint32_t numBuffers) {

        // Initialize particles
        time_t* t = nullptr;
        time(t);
        std::default_random_engine rndEngine(static_cast<unsigned int>(*t));
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
                    &m_Memory[i]))
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

    void createTextureTarget(uint32_t width, uint32_t height, uint32_t depth, uint32_t framesInFlight, uint32_t numTextures = 6) {
        m_TextureComputeTargets.resize(framesInFlight * numTextures);
        m_TextureComputeLeftInput.resize(framesInFlight);
        m_TextureComputeRightInput.resize(framesInFlight);
        m_TextureComputeTargets3D.resize(framesInFlight * 10);
        m_TextureDisparityTarget.resize(framesInFlight * 6);
        auto *array = new uint8_t[width * height * 2](); // The parentheses initialize all elements to zero.
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
        auto *array3D = new uint8_t[width * height * depth]; // The parentheses initialize all elements to zero.
        std::fill(array3D, array3D + width * height * depth , static_cast<uint8_t>(1));

        for (uint32_t i = 0; i < framesInFlight * 10; ++i) {

            m_TextureComputeTargets3D[i].fromBuffer(array3D, width * height * depth, VK_FORMAT_R8_UNORM, width, height,
                                                    depth,
                                                    m_VulkanDevice,
                                                    m_VulkanDevice->m_TransferQueue, VK_FILTER_LINEAR,
                                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL,
                                                    true);

        }
        for (uint32_t i = 0; i < framesInFlight * 4; ++i) {
        m_TextureDisparityTarget[i].fromBuffer(array, width * height * 2, VK_FORMAT_R16_UNORM, width, height,
                                               m_VulkanDevice,
                                               m_VulkanDevice->m_TransferQueue, VK_FILTER_LINEAR,
                                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                               VK_IMAGE_LAYOUT_GENERAL,
                                               true);
        }
        delete[] array;
        delete[] array3D;
    }


    void prepareDescriptors(uint32_t framesInFlight, const std::vector<VkRender::UniformBufferSet> &ubo,  std::vector<VkPipelineShaderStageCreateInfo> & shaders) {
        pass.resize(numPasses);
        descriptorLayoutProcessImagesPass();
        descriptorPoolProcessImagesPass(framesInFlight);
        descriptorSetsProcessImagePass(framesInFlight, ubo);
        for(uint32_t i = 0; i < numPasses; ++i){
            computePipelinePass(shaders[i], i);
        }
    }

    void descriptorLayoutProcessImagesPass() {
        std::array<VkDescriptorSetLayoutBinding, 9> layoutBindings{};
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (size_t i = 1; i < 9; ++i) {
            layoutBindings[i].binding = static_cast<uint32_t>(i);
            layoutBindings[i].descriptorCount = 1;
            layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            layoutBindings[i].pImmutableSamplers = nullptr;
            layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
        layoutInfo.pBindings = layoutBindings.data();
        if (vkCreateDescriptorSetLayout(m_VulkanDevice->m_LogicalDevice, &layoutInfo, nullptr,
                                        &pass[0].descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }



    void descriptorPoolProcessImagesPass(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2 * count},

        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(
                vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                       &pass[0].descriptorPool))
    }

    void descriptorSetsProcessImagePass(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        pass[0].descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, pass[0].descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pass[0].descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              pass[0].descriptors.data()))

        for (size_t i = 0; i < count; i++) {

            std::vector<VkWriteDescriptorSet> descriptorWrites(9);

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = pass[0].descriptors[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &ubo[i].bufferTwo.m_DescriptorBufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = pass[0].descriptors[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &m_TextureComputeLeftInput[i]->m_Descriptor;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = pass[0].descriptors[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pImageInfo = &m_TextureComputeRightInput[i]->m_Descriptor;

            for (size_t j = 3; j < 9; ++j) {
                descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[j].dstSet = pass[0].descriptors[i];
                descriptorWrites[j].dstBinding = static_cast<uint32_t>(j);
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[j].descriptorCount = 1;
                descriptorWrites[j].pImageInfo = &m_TextureComputeTargets[((j - 3) * count )+ i].m_Descriptor;
            }

            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }


    void computePipelinePass(VkPipelineShaderStageCreateInfo computeShaderStageInfo, size_t passNumber) {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &pass[passNumber].descriptorSetLayout;

        if (vkCreatePipelineLayout(m_VulkanDevice->m_LogicalDevice, &pipelineLayoutInfo, nullptr, &pass[passNumber].pipelineLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute m_Pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pass[passNumber].pipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;

        if (vkCreateComputePipelines(m_VulkanDevice->m_LogicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                     &pass[passNumber].pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute m_Pipeline!");
        }
    }


    void recordDrawCommands(CommandBuffer * commandBuffer, uint32_t currentFrame) {
        uint32_t workGroupSize = 24;

        vkCmdBindPipeline(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[0].pipeline);
        vkCmdBindDescriptorSets(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[0].pipelineLayout, 0, 1,
                                &pass[0].descriptors[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer->buffers[currentFrame], m_TextureComputeLeftInput[currentFrame]->m_Width / workGroupSize,
                      m_TextureComputeLeftInput[currentFrame]->m_Height / workGroupSize, 1);

    }

    ~ComputeShader() {

        for (auto &b: m_Buffer) {
            vkDestroyBuffer(m_VulkanDevice->m_LogicalDevice, b, nullptr);
        }
        for (auto &m: m_Memory) {
            vkFreeMemory(m_VulkanDevice->m_LogicalDevice, m, nullptr);
        }

        for (auto & pas : pass){
            vkDestroyPipelineLayout(m_VulkanDevice->m_LogicalDevice, pas.pipelineLayout, nullptr);
            vkDestroyPipeline(m_VulkanDevice->m_LogicalDevice, pas.pipeline, nullptr);
            vkDestroyDescriptorPool(m_VulkanDevice->m_LogicalDevice, pas.descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_VulkanDevice->m_LogicalDevice, pas.descriptorSetLayout, nullptr);
        }

    }

    std::vector<VkBuffer> m_Buffer{};
    std::vector<Texture2D> m_TextureComputeTargets{};
    std::vector<Texture2D> m_TextureDisparityTarget{};
    std::vector<Texture3D> m_TextureComputeTargets3D{};
    std::vector<std::unique_ptr<TextureVideo>> m_TextureComputeLeftInput{};
    std::vector<std::unique_ptr<TextureVideo>> m_TextureComputeRightInput{};

    std::vector<VkDeviceMemory> m_Memory{};


private:
    int PARTICLE_COUNT = 4096;
    size_t numPasses = 6;
    struct ComputePass {
        VkDescriptorSetLayout descriptorSetLayout{};
        VkDescriptorPool descriptorPool{};
        std::vector<VkDescriptorSet> descriptors{};
        VkPipeline pipeline{};
        VkPipelineLayout pipelineLayout{};
    };
    std::vector<ComputePass> pass;


};


#endif //MULTISENSE_VIEWER_COMPUTESHADER_H
