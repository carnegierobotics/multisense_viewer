//
// Created by magnus on 10/3/23.
//

#ifndef MULTISENSE_VIEWER_CUSTOMMODELS_H
#define MULTISENSE_VIEWER_CUSTOMMODELS_H


#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/Core/CommandBuffer.h"
#include "Viewer/VkRender/Components/RenderComponents/RenderBase.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/Components/Components.h"

namespace VkRender {
    struct CustomModelComponent : RenderBase {

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
            Log::Logger::getInstance()->trace("Loaded shader {} for stage {}", fileName, static_cast<uint32_t>(stage));
            return shaderStage;
        }

        void updateView(const Camera &camera);

        void updateTransform(const TransformComponent &transform);

        void draw(CommandBuffer *cmdBuffer) override;

        bool cleanUp(uint32_t currentFrame, bool force = false) override;

        void update(uint32_t currentFrame) override;

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

        const VulkanDevice *vulkanDevice = nullptr;
        const VkRender::RenderUtils *renderer;
        bool resourcesDeleted = false;

        std::unique_ptr<Model> model;
        std::vector<Buffer> UBOBuffers{};
        UBOMatrix mvp{};


        explicit CustomModelComponent(const VkRender::RenderUtils *renderUtils) {
            renderer = renderUtils;
            vulkanDevice = renderUtils->device;
            model = std::make_unique<Model>(renderUtils);

            UBOBuffers.resize(renderUtils->swapchainImages);

            for (auto &uniformBuffer: UBOBuffers) {
                renderUtils->device->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                  &uniformBuffer, sizeof(VkRender::UBOMatrix));
                uniformBuffer.map();
            }

            descriptors.resize(renderUtils->swapchainImages);
            descriptorSetLayouts.resize(renderUtils->swapchainImages);
            descriptorPools.resize(renderUtils->swapchainImages);
            pipelines.resize(renderUtils->swapchainImages);
            pipelineLayouts.resize(renderUtils->swapchainImages);


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
            //createGraphicsPipeline(shaders);

            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, vertModule, nullptr);
            vkDestroyShaderModule(vulkanDevice->m_LogicalDevice, fragModule, nullptr);
        }

        ~CustomModelComponent() override {
            if (!resourcesDeleted)
                cleanUp(0, true);


        }


        void createDescriptorSetLayout();

        void createDescriptorPool();

        void createDescriptorSets();

        void createGraphicsPipeline(std::vector<VkPipelineShaderStageCreateInfo> vector);

    };

};
#endif //MULTISENSE_VIEWER_CUSTOMMODELS_H
