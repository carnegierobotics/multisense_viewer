//
// Created by mgjer on 12/01/2024.
//

#ifndef MULTISENSE_VIEWER_RENDERRESOURCE_H
#define MULTISENSE_VIEWER_RENDERRESOURCE_H

#include <utility>
#include <vector>
#include <string>
#include <vulkan/vulkan_core.h>

#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Macros.h"

namespace RenderResource {
    struct Mesh {
    public:
        struct Data {
            VulkanDevice *device = nullptr;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            uint32_t vertexCount = 0;

            struct Vertices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};

                std::vector<VkRender::Vertex> data;
            } vertices{};
            struct Indices {
                VkBuffer buffer = VK_NULL_HANDLE;
                VkDeviceMemory memory{};

                std::vector<uint32_t> data;
            } indices{};

            Buffer uniformBuffer{};

        };

        struct Config {
            explicit Config(std::string _type = "default", VulkanDevice *_device = nullptr) : type(
                    std::move(_type)), device(_device) {

            }

            VulkanDevice *device = nullptr;
            std::string type;

        };
        static void defaultCube(RenderResource::Mesh::Data *mesh) {
            // Define the vertices of the cube

            std::vector<VkRender::Vertex> vertices{};
            vertices.resize(8);
            // Front face
            vertices[0].pos = glm::vec3(-0.5f, -0.5f, 0.5f); // 0
            vertices[1].pos = glm::vec3(0.5f, -0.5f, 0.5f); // 1
            vertices[2].pos = glm::vec3(0.5f, 0.5f, 0.5f); // 2
            vertices[3].pos = glm::vec3(-0.5f, 0.5f, 0.5f); // 3

            // Back face
            vertices[4].pos = glm::vec3(-0.5f, -0.5f, -0.5f); // 4
            vertices[5].pos = glm::vec3(0.5f, -0.5f, -0.5f); // 5
            vertices[6].pos = glm::vec3(0.5f, 0.5f, -0.5f); // 6
            vertices[7].pos = glm::vec3(-0.5f, 0.5f, -0.5f); // 7


            // Define the indices for the cube
            std::vector<uint32_t> indices = {
                    // Front face
                    0, 1, 2, 2, 3, 0,
                    // Right face
                    1, 5, 6, 6, 2, 1,
                    // Back face
                    7, 6, 5, 5, 4, 7,
                    // Left face
                    4, 0, 3, 3, 7, 4,
                    // Bottom face
                    4, 5, 1, 1, 0, 4,
                    // Top face
                    3, 2, 6, 6, 7, 3
            };

            mesh->firstIndex = 0;
            mesh->indexCount = indices.size();
            mesh->vertexCount = vertices.size();
            mesh->vertices.data = vertices;
            mesh->indices.data = indices;
        }

        static void uploadMeshDeviceLocal(RenderResource::Mesh::Data *mesh, VulkanDevice *vulkanDevice) {
            size_t vertexBufferSize = mesh->vertexCount * sizeof(VkRender::Vertex);
            size_t indexBufferSize = mesh->indexCount * sizeof(uint32_t);

            struct StagingBuffer {
                VkBuffer buffer;
                VkDeviceMemory memory;
            } vertexStaging{}, indexStaging{};

            // Create staging buffers
            // Vertex m_DataPtr
            CHECK_RESULT(vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    vertexBufferSize,
                    &vertexStaging.buffer,
                    &vertexStaging.memory,
                    reinterpret_cast<const void *>(mesh->vertices.data.data())))
            // Index m_DataPtr
            if (indexBufferSize > 0) {
                CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        indexBufferSize,
                        &indexStaging.buffer,
                        &indexStaging.memory,
                        reinterpret_cast<const void *>(mesh->indices.data.data())))
            }
            // Create m_Device local buffers
            // Vertex buffer
            if (mesh->vertices.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh->vertices.buffer, nullptr);
                vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh->vertices.memory, nullptr);
                if (indexBufferSize > 0) {
                    vkDestroyBuffer(vulkanDevice->m_LogicalDevice, mesh->indices.buffer, nullptr);
                    vkFreeMemory(vulkanDevice->m_LogicalDevice, mesh->indices.memory, nullptr);
                }
            }
            CHECK_RESULT(vulkanDevice->createBuffer(
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    vertexBufferSize,
                    &mesh->vertices.buffer,
                    &mesh->vertices.memory));
            // Index buffer
            if (indexBufferSize > 0) {
                CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        indexBufferSize,
                        &mesh->indices.buffer,
                        &mesh->indices.memory));
            }

            // Copy from staging buffers
            VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkBufferCopy copyRegion = {};
            copyRegion.size = vertexBufferSize;
            vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, mesh->vertices.buffer, 1, &copyRegion);
            if (indexBufferSize > 0) {
                copyRegion.size = indexBufferSize;
                vkCmdCopyBuffer(copyCmd, indexStaging.buffer, mesh->indices.buffer, 1, &copyRegion);
            }
            vulkanDevice->flushCommandBuffer(copyCmd, vulkanDevice->m_TransferQueue, true);
            vkDestroyBuffer(vulkanDevice->m_LogicalDevice, vertexStaging.buffer, nullptr);
            vkFreeMemory(vulkanDevice->m_LogicalDevice, vertexStaging.memory, nullptr);

            if (indexBufferSize > 0) {
                vkDestroyBuffer(vulkanDevice->m_LogicalDevice, indexStaging.buffer, nullptr);
                vkFreeMemory(vulkanDevice->m_LogicalDevice, indexStaging.memory, nullptr);
            }
        }

        static RenderResource::Mesh::Data createMesh(const RenderResource::Mesh::Config &config) {
            RenderResource::Mesh::Data mesh;
            defaultCube(&mesh);
            uploadMeshDeviceLocal(&mesh, config.device);

            return mesh;

        }
    };
}

namespace RenderResource {

    struct Pipeline {
        struct Config {
            explicit Config(std::string _type = "default", VulkanDevice *_device = nullptr) : type(
                    std::move(_type)), device(_device) {

            }

            VulkanDevice *device = nullptr;
            std::string type;
            uint32_t UboCount = 0;
            VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
            VkRenderPass *renderPass{};
            Buffer* ubo{};
            std::vector<VkPipelineShaderStageCreateInfo> shaders;
        };

        struct Data{
            std::vector<VkDescriptorSet> descriptors;
            VkDescriptorSetLayout descriptorSetLayout{};
            VkDescriptorPool descriptorPool{};
            VkPipeline pipeline{};
            bool initializedPipeline = false;
            VkPipelineLayout pipelineLayout{};

        };

        static void createDefaultDescriptorLayout(Data *pipeline, const Config *conf) {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};

            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            };

            VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(setLayoutBindings.data(),
                                                                                                       static_cast<uint32_t>(setLayoutBindings.size()));
            CHECK_RESULT(vkCreateDescriptorSetLayout(conf->device->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                                     &pipeline->descriptorSetLayout));
        }

        static void createDefaultDescriptorPool(Data *pipeline, const Config *conf) {

            uint32_t uniformDescriptorCount = conf->UboCount;
            std::vector<VkDescriptorPoolSize> poolSizes = {
                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformDescriptorCount},
            };
            VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, conf->UboCount);

            CHECK_RESULT(
                    vkCreateDescriptorPool(conf->device->m_LogicalDevice, &poolCreateInfo, nullptr, &pipeline->descriptorPool));
        }
        static void createDescriptorSets(Data *pipeline, const Config *conf) {

            pipeline->descriptors.resize(conf->UboCount);
            for (size_t i = 0; i < conf->UboCount; i++) {

                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = pipeline->descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &pipeline->descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                CHECK_RESULT(vkAllocateDescriptorSets(conf->device->m_LogicalDevice, &descriptorSetAllocInfo,
                                                      &pipeline->descriptors[i]));

                std::vector<VkWriteDescriptorSet> writeDescriptorSets(1);
                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].dstSet = pipeline->descriptors[i];
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].pBufferInfo = &conf->ubo[i].m_DescriptorBufferInfo;


                vkUpdateDescriptorSets(conf->device->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, nullptr);
            }
        }

        static void createGraphicsPipeline(Data *pipeline, const Config *conf) {
            VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&pipeline->descriptorSetLayout, 1);
            CHECK_RESULT(vkCreatePipelineLayout(conf->device->m_LogicalDevice, &info, nullptr, &pipeline->pipelineLayout))

            // Vertex bindings an attributes
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
            inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
            rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
            rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterizationStateCI.lineWidth = 1.0f;

            VkPipelineColorBlendAttachmentState blendAttachmentState{};
            blendAttachmentState.colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blendAttachmentState.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
            colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlendStateCI.attachmentCount = 1;
            colorBlendStateCI.pAttachments = &blendAttachmentState;

            VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
            depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencilStateCI.depthTestEnable = VK_TRUE;
            depthStencilStateCI.depthWriteEnable = VK_TRUE;
            depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS;
            depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
            depthStencilStateCI.stencilTestEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo viewportStateCI{};
            viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportStateCI.viewportCount = 1;
            viewportStateCI.scissorCount = 1;

            VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
            multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampleStateCI.rasterizationSamples = conf->msaaSamples;


            std::vector<VkDynamicState> dynamicStateEnables = {
                    VK_DYNAMIC_STATE_VIEWPORT,
                    VK_DYNAMIC_STATE_SCISSOR
            };
            VkPipelineDynamicStateCreateInfo dynamicStateCI{};
            dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicStateCI.pDynamicStates = dynamicStateEnables.data();
            dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());


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
            VkGraphicsPipelineCreateInfo pipelineCI{};
            pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCI.layout = pipeline->pipelineLayout;
            pipelineCI.renderPass = *conf->renderPass;
            pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
            pipelineCI.pVertexInputState = &vertexInputStateCI;
            pipelineCI.pRasterizationState = &rasterizationStateCI;
            pipelineCI.pColorBlendState = &colorBlendStateCI;
            pipelineCI.pMultisampleState = &multisampleStateCI;
            pipelineCI.pViewportState = &viewportStateCI;
            pipelineCI.pDepthStencilState = &depthStencilStateCI;
            pipelineCI.pDynamicState = &dynamicStateCI;
            pipelineCI.stageCount = static_cast<uint32_t>(conf->shaders.size());
            pipelineCI.pStages = conf->shaders.data();

            VkResult res = vkCreateGraphicsPipelines(conf->device->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                                     &pipeline->pipeline);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create graphics m_Pipeline");

        }

        static RenderResource::Pipeline::Data createRenderPipeline(const RenderResource::Pipeline::Config& config){
            Data data;
            createDefaultDescriptorLayout(&data, &config);
            createDefaultDescriptorPool(&data, &config);
            createDescriptorSets(&data, &config);
            createGraphicsPipeline(&data, &config);

            return data;
        }
    };
}

#endif //MULTISENSE_VIEWER_RENDERRESOURCE_H
