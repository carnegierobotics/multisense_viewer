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
#include "RenderDefinitions.h"

namespace RenderResource {

    struct MeshConfig {
        VulkanDevice *device = nullptr;
        std::string type;

    };

    class Mesh {
    public:
        explicit Mesh(const RenderResource::MeshConfig &meshConf) : conf(meshConf) {
            device = meshConf.device;
            defaultCube(&model);
            uploadMeshDeviceLocal(&model);
        }

        ~Mesh() {
            if (model.vertices.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device->m_LogicalDevice, model.vertices.buffer, nullptr);
                vkFreeMemory(device->m_LogicalDevice, model.vertices.memory, nullptr);
                if (model.indexCount > 0) {
                    vkDestroyBuffer(device->m_LogicalDevice, model.indices.buffer, nullptr);
                    vkFreeMemory(device->m_LogicalDevice, model.indices.memory, nullptr);
                }
            }
        }

        VulkanDevice *device = nullptr;
        MeshConfig conf;

        struct Model {
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


        } model;

    private:
        void defaultCube(RenderResource::Mesh::Model *mesh) {
            // Define the vertices of the cube

            std::vector<VkRender::Vertex> vertices{};
            vertices.resize(4);
            // Front face
            vertices[0].pos = glm::vec3(-1.0f, -1.0f, 0.0f); // Bottom-left Top-left
            vertices[0].uv0 = glm::vec2(0.0f, 0.0f);

            vertices[1].pos = glm::vec3(1.0f, -1.0f, 0.0f); // Bottom-right
            vertices[1].uv0 = glm::vec2(1.0f, 0.0f);

            vertices[2].pos = glm::vec3(1.0f, 1.0f, 0.0f); // Top-right
            vertices[2].uv0 = glm::vec2(1.0f, 1.0f);

            vertices[3].pos = glm::vec3(-1.0f, 1.0f, 0.0f); // Top-left
            vertices[3].uv0 = glm::vec2(0.0f, 1.0f);


            // Define the indices for the cube
            std::vector<uint32_t> indices = {
                    0, 1, 2, // First triangle (bottom-right)
                    2, 3, 0  // Second triangle (top-left)
            };

            mesh->firstIndex = 0;
            mesh->indexCount = static_cast<uint32_t>(indices.size());
            mesh->vertexCount = static_cast<uint32_t>(vertices.size());
            mesh->vertices.data = vertices;
            mesh->indices.data = indices;
        }

        void uploadMeshDeviceLocal(RenderResource::Mesh::Model *mesh) {
            size_t vertexBufferSize = mesh->vertexCount * sizeof(VkRender::Vertex);
            size_t indexBufferSize = mesh->indexCount * sizeof(uint32_t);

            struct StagingBuffer {
                VkBuffer buffer;
                VkDeviceMemory memory;
            } vertexStaging{}, indexStaging{};

            // Create staging buffers
            // Vertex m_DataPtr
            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    vertexBufferSize,
                    &vertexStaging.buffer,
                    &vertexStaging.memory,
                    reinterpret_cast<const void *>(mesh->vertices.data.data())))
            // Index m_DataPtr
            if (indexBufferSize > 0) {
                CHECK_RESULT(device->createBuffer(
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
                vkDestroyBuffer(device->m_LogicalDevice, mesh->vertices.buffer, nullptr);
                vkFreeMemory(device->m_LogicalDevice, mesh->vertices.memory, nullptr);
                if (indexBufferSize > 0) {
                    vkDestroyBuffer(device->m_LogicalDevice, mesh->indices.buffer, nullptr);
                    vkFreeMemory(device->m_LogicalDevice, mesh->indices.memory, nullptr);
                }
            }
            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    vertexBufferSize,
                    &mesh->vertices.buffer,
                    &mesh->vertices.memory));
            // Index buffer
            if (indexBufferSize > 0) {
                CHECK_RESULT(device->createBuffer(
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        indexBufferSize,
                        &mesh->indices.buffer,
                        &mesh->indices.memory));
            }

            // Copy from staging buffers
            VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkBufferCopy copyRegion = {};
            copyRegion.size = vertexBufferSize;
            vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, mesh->vertices.buffer, 1, &copyRegion);
            if (indexBufferSize > 0) {
                copyRegion.size = indexBufferSize;
                vkCmdCopyBuffer(copyCmd, indexStaging.buffer, mesh->indices.buffer, 1, &copyRegion);
            }
            device->flushCommandBuffer(copyCmd, device->m_TransferQueue, true);
            vkDestroyBuffer(device->m_LogicalDevice, vertexStaging.buffer, nullptr);
            vkFreeMemory(device->m_LogicalDevice, vertexStaging.memory, nullptr);

            if (indexBufferSize > 0) {
                vkDestroyBuffer(device->m_LogicalDevice, indexStaging.buffer, nullptr);
                vkFreeMemory(device->m_LogicalDevice, indexStaging.memory, nullptr);
            }
        }
    };
}

namespace RenderResource {
    struct PipelineConfig {
        VulkanDevice *device = nullptr;
        uint32_t UboCount = 0;
        VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
        VkRenderPass *renderPass{};
        Buffer *ubo{};
        std::vector<VkPipelineShaderStageCreateInfo> *shaders{};
        std::vector<Texture2D> *textures{};
    };

    class Pipeline {
    public:
        VulkanDevice *device = nullptr;
        struct Data {
            std::vector<VkDescriptorSet> descriptors;
            VkDescriptorSetLayout descriptorSetLayout{};
            VkDescriptorPool descriptorPool{};
            VkPipeline pipeline{};
            VkPipelineLayout pipelineLayout{};
        } data;

        explicit Pipeline(PipelineConfig _conf) : conf(_conf) {
            device = _conf.device;

            createDefaultDescriptorLayout();
            createDefaultDescriptorPool();
            createDescriptorSets();
            createGraphicsPipeline();
        }

        ~Pipeline() {
            vkDestroyDescriptorSetLayout(device->m_LogicalDevice, data.descriptorSetLayout, nullptr);
            vkDestroyDescriptorPool(device->m_LogicalDevice, data.descriptorPool, nullptr);
            vkDestroyPipelineLayout(device->m_LogicalDevice, data.pipelineLayout, nullptr);
            vkDestroyPipeline(device->m_LogicalDevice, data.pipeline, nullptr);
        }

        PipelineConfig conf;

        void createDefaultDescriptorLayout() {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};

            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT,   nullptr},
                    {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };

            VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                    setLayoutBindings.data(),
                    static_cast<uint32_t>(setLayoutBindings.size()));
            CHECK_RESULT(vkCreateDescriptorSetLayout(device->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                                     &data.descriptorSetLayout));
        }

        void createDefaultDescriptorPool() {

            uint32_t framesInFlight = conf.UboCount;
            std::vector<VkDescriptorPoolSize> poolSizes = {
                    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         framesInFlight},
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, framesInFlight},
            };
            VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, framesInFlight *
                    static_cast<uint32_t>(poolSizes.size()));

            CHECK_RESULT(
                    vkCreateDescriptorPool(device->m_LogicalDevice, &poolCreateInfo, nullptr,
                                           &data.descriptorPool));
        }

        void createDescriptorSets() {

            data.descriptors.resize(conf.UboCount);
            for (size_t i = 0; i < conf.UboCount; i++) {

                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = data.descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &data.descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                CHECK_RESULT(vkAllocateDescriptorSets(device->m_LogicalDevice, &descriptorSetAllocInfo,
                                                      &data.descriptors[i]));

                std::vector<VkWriteDescriptorSet> writeDescriptorSets(2);
                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].dstSet = data.descriptors[i];
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].pBufferInfo = &conf.ubo[i].m_DescriptorBufferInfo;

                writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[1].descriptorCount = 1;
                writeDescriptorSets[1].dstSet = data.descriptors[i];
                writeDescriptorSets[1].dstBinding = 1;
                writeDescriptorSets[1].pImageInfo = &(*conf.textures)[i].m_Descriptor;

                vkUpdateDescriptorSets(device->m_LogicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, nullptr);
            }
        }

        void createGraphicsPipeline() {
            VkPipelineLayoutCreateInfo info = Populate::pipelineLayoutCreateInfo(&data.descriptorSetLayout, 1);
            CHECK_RESULT(
                    vkCreatePipelineLayout(device->m_LogicalDevice, &info, nullptr, &data.pipelineLayout))

            // Vertex bindings an attributes
            VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
            inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssemblyStateCI.primitiveRestartEnable = VK_FALSE;

            VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
            rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizationStateCI.cullMode = VK_CULL_MODE_NONE;
            rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterizationStateCI.lineWidth = 1.0f;
            rasterizationStateCI.depthClampEnable = VK_FALSE;
            rasterizationStateCI.rasterizerDiscardEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState blendAttachmentState{};
            blendAttachmentState.colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                    VK_COLOR_COMPONENT_A_BIT;
            blendAttachmentState.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
            colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlendStateCI.logicOpEnable = VK_FALSE;
            colorBlendStateCI.logicOp = VK_LOGIC_OP_COPY;
            colorBlendStateCI.attachmentCount = 1;
            colorBlendStateCI.pAttachments = &blendAttachmentState;
            colorBlendStateCI.blendConstants[0] = 0.0f;
            colorBlendStateCI.blendConstants[1] = 0.0f;
            colorBlendStateCI.blendConstants[2] = 0.0f;
            colorBlendStateCI.blendConstants[3] = 0.0f;

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
            multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


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
            VkGraphicsPipelineCreateInfo pipelineCI{};
            pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineCI.layout = data.pipelineLayout;
            pipelineCI.renderPass = *conf.renderPass;
            pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
            pipelineCI.pVertexInputState = &vertexInputStateCI;
            pipelineCI.pRasterizationState = &rasterizationStateCI;
            pipelineCI.pColorBlendState = &colorBlendStateCI;
            pipelineCI.pMultisampleState = &multisampleStateCI;
            pipelineCI.pViewportState = &viewportStateCI;
            pipelineCI.pDepthStencilState = &depthStencilStateCI;
            pipelineCI.pDynamicState = &dynamicStateCI;
            pipelineCI.stageCount = static_cast<uint32_t>((*conf.shaders).size());
            pipelineCI.pStages = (*conf.shaders).data();

            VkResult res = vkCreateGraphicsPipelines(device->m_LogicalDevice, nullptr, 1, &pipelineCI, nullptr,
                                                     &data.pipeline);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create graphics m_Pipeline");
        }
    };
}

#endif //MULTISENSE_VIEWER_RENDERRESOURCE_H
