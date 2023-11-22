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
#include "Viewer/Core/Definitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Macros.h"
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

    void createTextureTarget(uint32_t width, uint32_t height, uint32_t depth, uint32_t framesInFlight, uint32_t numTextures = 6) {
        m_TextureComputeTargets.resize(framesInFlight * numTextures);
        m_TextureComputeLeftInput.resize(framesInFlight);
        m_TextureComputeRightInput.resize(framesInFlight);
        m_TextureComputeTargets3D.resize(framesInFlight * 10);
        m_TextureDisparityTarget.resize(framesInFlight * 6);
        uint8_t *array = new uint8_t[width * height * 2](); // The parentheses initialize all elements to zero.
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
        uint8_t *array3D = new uint8_t[width * height * depth]; // The parentheses initialize all elements to zero.
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
        descriptorLayoutPixelCostPass();
        descriptorLayoutWindowedPixelCostPass();
        descriptorLayoutProcessPathPass();
        descriptorLayoutCalcDispPass();
        descriptorLayoutLRCheckPass();

        descriptorPoolProcessImagesPass(framesInFlight);
        descriptorPoolPixelCostPass(framesInFlight);
        descriptorPoolWindowedPixelCostPass(framesInFlight);
        descriptorPoolProcessImagePath(framesInFlight);
        descriptorPoolCalcDispPass(framesInFlight);
        descriptorPoolLRCheckPass(framesInFlight);

        descriptorSetsProcessImagePass(framesInFlight, ubo);
        descriptorSetsPixelCostPass(framesInFlight, ubo);
        descriptorSetsWindowedPixelCostPass(framesInFlight, ubo);
        descriptorSetsProcessPathPass(framesInFlight, ubo);
        descriptorSetsCalcDispPass(framesInFlight, ubo);
        descriptorSetsLRCheckPass(framesInFlight, ubo);

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
            layoutBindings[i].binding = i;
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

    void descriptorLayoutPixelCostPass() {
        std::array<VkDescriptorSetLayoutBinding, 7> layoutBindings{};

        for (size_t i = 0; i < layoutBindings.size(); ++i) {
            layoutBindings[i].binding = i;
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
                                        &pass[1].descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }
 void descriptorLayoutWindowedPixelCostPass() {
        std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

        for (size_t i = 0; i < layoutBindings.size(); ++i) {
            layoutBindings[i].binding = i;
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
                                        &pass[2].descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void descriptorLayoutProcessPathPass() {
        std::array<VkDescriptorSetLayoutBinding, 9> layoutBindings{};
        for (size_t i = 0; i < layoutBindings.size(); ++i) {
            layoutBindings[i].binding = i;
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
                                        &pass[3].descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void descriptorLayoutCalcDispPass() {
        std::array<VkDescriptorSetLayoutBinding, 7> layoutBindings{};
        for (size_t i = 0; i < layoutBindings.size(); ++i) {
            layoutBindings[i].binding = i;
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
                                        &pass[4].descriptorSetLayout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void descriptorLayoutLRCheckPass() {
        std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};
        for (size_t i = 0; i < layoutBindings.size(); ++i) {
            layoutBindings[i].binding = i;
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
                                        &pass[5].descriptorSetLayout) !=
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
                                       &pass[0].descriptorPool));
    }

    void descriptorPoolPixelCostPass(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 * count},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 * count},
        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                            &pass[1].descriptorPool));
    }
    void descriptorPoolWindowedPixelCostPass(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * count},
        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                            &pass[2].descriptorPool));
    }

    void descriptorPoolProcessImagePath(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9 * count},
        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                            &pass[3].descriptorPool));
    }

    void descriptorPoolCalcDispPass(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 * count},
        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                            &pass[4].descriptorPool));
    }

    void descriptorPoolLRCheckPass(uint32_t count) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * count},
        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, count);
        CHECK_RESULT(vkCreateDescriptorPool(m_VulkanDevice->m_LogicalDevice, &poolCreateInfo, nullptr,
                                            &pass[5].descriptorPool));
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
                                              pass[0].descriptors.data()));

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
                descriptorWrites[j].dstBinding = j;
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[j].descriptorCount = 1;
                descriptorWrites[j].pImageInfo = &m_TextureComputeTargets[((j - 3) * count )+ i].m_Descriptor;
            }

            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }

    void descriptorSetsPixelCostPass(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        pass[1].descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, pass[1].descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pass[1].descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              pass[1].descriptors.data()));

        for (size_t i = 0; i < count; i++) {

            std::vector<VkWriteDescriptorSet> descriptorWrites(7);
            for (size_t j = 0; j < 6; ++j) {
                descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[j].dstSet = pass[1].descriptors[i];
                descriptorWrites[j].dstBinding = j;
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[j].descriptorCount = 1;
                descriptorWrites[j].pImageInfo = &m_TextureComputeTargets[i + (j * count)].m_Descriptor;
            }

            descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[6].dstSet = pass[1].descriptors[i];
            descriptorWrites[6].dstBinding = 6;
            descriptorWrites[6].dstArrayElement = 0;
            descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[6].descriptorCount = 1;
            descriptorWrites[6].pImageInfo = &m_TextureComputeTargets3D[i].m_Descriptor;

            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }

    void descriptorSetsWindowedPixelCostPass(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        pass[2].descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, pass[2].descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pass[2].descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              pass[2].descriptors.data()));

        for (size_t i = 0; i < count; i++) {
            std::vector<VkWriteDescriptorSet> descriptorWrites(2);
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = pass[2].descriptors[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &m_TextureComputeTargets3D[i].m_Descriptor;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = pass[2].descriptors[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &m_TextureComputeTargets3D[i + count].m_Descriptor;


            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }

    void descriptorSetsProcessPathPass(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        pass[3].descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, pass[3].descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pass[3].descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              pass[3].descriptors.data()));

        for (size_t i = 0; i < count; i++) {

            std::vector<VkWriteDescriptorSet> descriptorWrites(9);
            for (size_t j = 0; j < descriptorWrites.size(); ++j) {
                descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[j].dstSet = pass[3].descriptors[i];
                descriptorWrites[j].dstBinding = j;
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[j].descriptorCount = 1;
                size_t idx = count + i + (j * count);
                descriptorWrites[j].pImageInfo = &m_TextureComputeTargets3D[idx].m_Descriptor;
            }

            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }

    void descriptorSetsCalcDispPass(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        pass[4].descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, pass[4].descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pass[4].descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              pass[4].descriptors.data()));

        for (size_t i = 0; i < count; i++) {

            std::vector<VkWriteDescriptorSet> descriptorWrites(7);
            for (size_t j = 0; j < descriptorWrites.size() - 1; ++j) {
                descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[j].dstSet = pass[4].descriptors[i];
                descriptorWrites[j].dstBinding = j;
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[j].descriptorCount = 1;
                size_t idx = (count * 2) + i + (j * count);
                descriptorWrites[j].pImageInfo = &m_TextureComputeTargets3D[idx].m_Descriptor;
            }

            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = pass[4].descriptors[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pImageInfo = &m_TextureDisparityTarget[i].m_Descriptor;

            descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[5].dstSet = pass[4].descriptors[i];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pImageInfo = &m_TextureDisparityTarget[i + count].m_Descriptor;

            descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[6].dstSet = pass[4].descriptors[i];
            descriptorWrites[6].dstBinding = 6;
            descriptorWrites[6].dstArrayElement = 0;
            descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[6].descriptorCount = 1;
            descriptorWrites[6].pImageInfo = &m_TextureDisparityTarget[i + (count * 3)].m_Descriptor;

            vkUpdateDescriptorSets(m_VulkanDevice->m_LogicalDevice, static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }

    void descriptorSetsLRCheckPass(uint32_t count, const std::vector<VkRender::UniformBufferSet> &ubo) {
        pass[5].descriptors.resize(count);
        std::vector<VkDescriptorSetLayout> layouts(count, pass[5].descriptorSetLayout);

        VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
        descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocInfo.descriptorPool = pass[5].descriptorPool;
        descriptorSetAllocInfo.pSetLayouts = layouts.data();
        descriptorSetAllocInfo.descriptorSetCount = count;
        CHECK_RESULT(vkAllocateDescriptorSets(m_VulkanDevice->m_LogicalDevice, &descriptorSetAllocInfo,
                                              pass[5].descriptors.data()));

        for (size_t i = 0; i < count; i++) {
            std::vector<VkWriteDescriptorSet> descriptorWrites(3);
            for (size_t j = 0; j < descriptorWrites.size(); ++j) {
                descriptorWrites[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[j].dstSet = pass[5].descriptors[i];
                descriptorWrites[j].dstBinding = j;
                descriptorWrites[j].dstArrayElement = 0;
                descriptorWrites[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[j].descriptorCount = 1;
                size_t idx = i + (j * count);
                descriptorWrites[j].pImageInfo = &m_TextureDisparityTarget[idx].m_Descriptor;
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

        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;  // Layout of the image before the barrier
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;  // Layout of the image after the barrier
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // Operations to wait on and stages where those operations occur
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; // Queue family ownership transfer (not used here)
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_TextureComputeTargets[(pass[0].descriptors.size() * 5) + currentFrame].m_Image; // Image being accessed and modified as part of the barrier
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // Aspect of the image being altered
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        // Insert the barrier
        vkCmdPipelineBarrier(
                commandBuffer->buffers[currentFrame],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Pipeline stages that the barrier is inserted between
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // Dependency flags
                0, nullptr, // Memory barriers
                0, nullptr, // Buffer memory barriers
                1, &barrier // Image memory barriers
        );

        vkCmdBindPipeline(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[1].pipeline);

        vkCmdBindDescriptorSets(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[1].pipelineLayout, 0, 1,
                                &pass[1].descriptors[currentFrame], 0, nullptr);

        vkCmdDispatch(commandBuffer->buffers[currentFrame], m_TextureComputeLeftInput[currentFrame]->m_Width / workGroupSize,
                      m_TextureComputeLeftInput[currentFrame]->m_Height / workGroupSize, 1);


        barrier.image = m_TextureComputeTargets3D[currentFrame].m_Image; // Image being accessed and modified as part of the barrier
        // Insert the barrier
        vkCmdPipelineBarrier(
                commandBuffer->buffers[currentFrame],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Pipeline stages that the barrier is inserted between
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // Dependency flags
                0, nullptr, // Memory barriers
                0, nullptr, // Buffer memory barriers
                1, &barrier // Image memory barriers
        );


        vkCmdBindPipeline(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[2].pipeline);

        vkCmdBindDescriptorSets(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[2].pipelineLayout, 0, 1,
                                &pass[2].descriptors[currentFrame], 0, nullptr);

        vkCmdDispatch(commandBuffer->buffers[currentFrame], m_TextureComputeLeftInput[currentFrame]->m_Width / workGroupSize,
                      m_TextureComputeLeftInput[currentFrame]->m_Height / workGroupSize, 1);


        barrier.image = m_TextureComputeTargets3D[currentFrame + pass[0].descriptors.size()].m_Image; // Image being accessed and modified as part of the barrier
        // Insert the barrier
        vkCmdPipelineBarrier(
                commandBuffer->buffers[currentFrame],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Pipeline stages that the barrier is inserted between
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // Dependency flags
                0, nullptr, // Memory barriers
                0, nullptr, // Buffer memory barriers
                1, &barrier // Image memory barriers
        );

        vkCmdBindPipeline(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[3].pipeline);
        vkCmdBindDescriptorSets(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[3].pipelineLayout, 0, 1,
                                &pass[3].descriptors[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer->buffers[currentFrame], 960 / workGroupSize, 600 / workGroupSize, 1);

        barrier.image = m_TextureComputeTargets3D[(pass[3].descriptors.size() * 4) + currentFrame].m_Image; // Image being accessed and modified as part of the barrier
        // Insert the barrier
        vkCmdPipelineBarrier(
                commandBuffer->buffers[currentFrame],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Pipeline stages that the barrier is inserted between
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // Dependency flags
                0, nullptr, // Memory barriers
                0, nullptr, // Buffer memory barriers
                1, &barrier // Image memory barriers
        );

        vkCmdBindPipeline(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[4].pipeline);
        vkCmdBindDescriptorSets(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[4].pipelineLayout, 0, 1,
                                &pass[4].descriptors[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer->buffers[currentFrame], m_TextureComputeLeftInput[currentFrame]->m_Width / workGroupSize,
                      m_TextureComputeLeftInput[currentFrame]->m_Height / workGroupSize, 1);


        barrier.image = m_TextureDisparityTarget[(pass[4].descriptors.size() * 2) + currentFrame].m_Image; // Image being accessed and modified as part of the barrier
        // Insert the barrier
        vkCmdPipelineBarrier(
                commandBuffer->buffers[currentFrame],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Pipeline stages that the barrier is inserted between
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // Dependency flags
                0, nullptr, // Memory barriers
                0, nullptr, // Buffer memory barriers
                1, &barrier // Image memory barriers
        );

        vkCmdBindPipeline(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[5].pipeline);
        vkCmdBindDescriptorSets(commandBuffer->buffers[currentFrame], VK_PIPELINE_BIND_POINT_COMPUTE, pass[5].pipelineLayout, 0, 1,
                                &pass[5].descriptors[currentFrame], 0, nullptr);
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

        for (size_t i = 0; i < pass.size(); ++i){
            vkDestroyPipelineLayout(m_VulkanDevice->m_LogicalDevice, pass[i].pipelineLayout, nullptr);
            vkDestroyPipeline(m_VulkanDevice->m_LogicalDevice, pass[i].pipeline, nullptr);
            vkDestroyDescriptorPool(m_VulkanDevice->m_LogicalDevice, pass[i].descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_VulkanDevice->m_LogicalDevice, pass[i].descriptorSetLayout, nullptr);
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
