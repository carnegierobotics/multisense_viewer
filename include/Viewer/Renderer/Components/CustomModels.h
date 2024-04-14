//
// Created by magnus on 10/3/23.
//

#ifndef MULTISENSE_VIEWER_CUSTOMMODELS_H
#define MULTISENSE_VIEWER_CUSTOMMODELS_H


#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Scripts/Private/TextureDataDef.h"
#include "Viewer/Core/CommandBuffer.h"

namespace VkRender {
    struct CustomModelComponent {

        static VkPipelineShaderStageCreateInfo
        loadShader(VkDevice device, std::string fileName, VkShaderStageFlagBits stage, VkShaderModule *module) {
            // Check if we have .spv extensions. If not then add it.
            std::size_t extension = fileName.find(".spv");
            if (extension == std::string::npos)
                fileName.append(".spv");
            Utils::loadShader((Utils::getShadersPath().append(fileName)).string().c_str(),
                              device, module);
            assert(module != VK_NULL_HANDLE);
            VkPipelineShaderStageCreateInfo shaderStage = {};
            shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStage.stage = stage;
            shaderStage.module = *module;
            shaderStage.pName = "main";
            Log::Logger::getInstance()->info("Loaded shader {} for stage {}", fileName, static_cast<uint32_t>(stage));
            return shaderStage;
        }


        struct Model {
            explicit Model(const VkRender::RenderUtils *renderUtils);

            ~Model();

            /**@brief Property to flashing/disable drawing of this m_Model. Set to false if you want to control when to draw the m_Model. */
            bool draw = true;

            struct Mesh {
                VulkanDevice *device = nullptr;
                uint32_t firstIndex = 0;
                uint32_t indexCount = 0;
                uint32_t vertexCount = 0;

                struct Vertices {
                    VkBuffer buffer = VK_NULL_HANDLE;
                    VkDeviceMemory memory{};
                } vertices{};

                struct Indices {
                    VkBuffer buffer = VK_NULL_HANDLE;
                    VkDeviceMemory memory{};
                } indices{};

                Buffer uniformBuffer{};
            } mesh{};

            struct Dimensions {
                glm::vec3 min = glm::vec3(FLT_MAX);
                glm::vec3 max = glm::vec3(-FLT_MAX);
            } dimensions;

            VulkanDevice *vulkanDevice{};
            std::vector<std::string> extensions;
            std::vector<Texture::TextureSampler> textureSamplers;

            void uploadMeshDeviceLocal(const std::vector<VkRender::Vertex> &vertices,
                                       const std::vector<uint32_t> &indices = std::vector<uint32_t>());
        };

    public:
        std::vector<VkDescriptorSet> descriptors{};
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts{};
        std::vector<VkDescriptorPool> descriptorPools{};
        std::vector<VkPipeline> pipelines{};
        std::vector<VkPipelineLayout> pipelineLayouts{};
        std::vector<bool> resourcesInUse = {false};

        bool markedForDeletion = false;
        const VulkanDevice *vulkanDevice = nullptr;
        const VkRender::RenderUtils *renderer;

        std::unique_ptr<Model> model;
        std::vector<Buffer> UBOBuffers{};

        void update(uint32_t index, void *data) {

            Buffer &currentUB = UBOBuffers[index];
            memcpy(currentUB.mapped, data, sizeof(VkRender::UBOMatrix));

        }

        explicit CustomModelComponent(const VkRender::RenderUtils *renderUtils) {
            renderer = renderUtils;
            vulkanDevice = renderUtils->device;
            model = std::make_unique<Model>(renderUtils);

            UBOBuffers.resize(renderUtils->UBCount);

            for (auto &uniformBuffer: UBOBuffers) {
                renderUtils->device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                  &uniformBuffer, sizeof(VkRender::UBOMatrix));
                uniformBuffer.map();
            }

            descriptors.resize(renderUtils->UBCount);
            descriptorSetLayouts.resize(renderUtils->UBCount);
            descriptorPools.resize(renderUtils->UBCount);
            pipelines.resize(renderUtils->UBCount);
            pipelineLayouts.resize(renderUtils->UBCount);


            VkShaderModule vertModule, fragModule;

            std::vector<VkPipelineShaderStageCreateInfo> shaders = {{CustomModelComponent::loadShader(
                    vulkanDevice->m_LogicalDevice, "spv/grid.vert",
                    VK_SHADER_STAGE_VERTEX_BIT, &vertModule)},
                                                                    {CustomModelComponent::loadShader(
                                                                            vulkanDevice->m_LogicalDevice,
                                                                            "spv/grid.frag",
                                                                            VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                            &fragModule)}};


            createDescriptorSetLayout();
            createDescriptorPool();
            createDescriptorSets();
            createGraphicsPipeline(shaders);

            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, vertModule, nullptr);
            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, fragModule, nullptr);
        }

        ~CustomModelComponent() {

            for (size_t i = 0; i < renderer->UBCount; ++i) {
                vkDestroyPipeline(renderer->device->m_LogicalDevice, pipelines[i], nullptr);
                vkDestroyPipelineLayout(renderer->device->m_LogicalDevice, pipelineLayouts[i], nullptr);
                vkDestroyDescriptorSetLayout(renderer->device->m_LogicalDevice, descriptorSetLayouts[i], nullptr);
                vkDestroyDescriptorPool(renderer->device->m_LogicalDevice, descriptorPools[i], nullptr);

            }

        }


        void createDescriptorSetLayout();

        void createDescriptorPool();

        void createDescriptorSets();

        void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> vector);

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex);
    };

};
#endif //MULTISENSE_VIEWER_CUSTOMMODELS_H
