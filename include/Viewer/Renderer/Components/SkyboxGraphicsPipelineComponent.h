//
// Created by mgjer on 12/01/2024.
//

#ifndef MULTISENSE_VIEWER_SKYBOXGRAPHICSPIPELINECOMPONENT_H
#define MULTISENSE_VIEWER_SKYBOXGRAPHICSPIPELINECOMPONENT_H

#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan_core.h>

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Core/CommandBuffer.h"

namespace RenderResource {


    struct SkyboxGraphicsPipelineComponent {

        SkyboxGraphicsPipelineComponent() = default;

        SkyboxGraphicsPipelineComponent(const SkyboxGraphicsPipelineComponent &) = default;

        SkyboxGraphicsPipelineComponent &operator=(const SkyboxGraphicsPipelineComponent &other) { return *this; }

        SkyboxGraphicsPipelineComponent(VkRender::RenderUtils *utils,
                                        const VkRender::GLTFModelComponent &modelComponent) {
            renderUtils = utils;
            vulkanDevice = utils->device;


            textures.empty.fromKtxFile(Utils::getTexturePath() / "empty.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, vulkanDevice->m_TransferQueue);
            textures.environmentCube.fromKtxFile(Utils::getTexturePath() / "Environments" / "skies.ktx2", vulkanDevice);

            generateCubemaps(modelComponent);
            generateBRDFLUT();
            setupUniformBuffers();
            setupDescriptors(modelComponent);
            setupPipelines();
        }

        ~SkyboxGraphicsPipelineComponent(){

            vkDestroyDescriptorPool(vulkanDevice->m_LogicalDevice, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(vulkanDevice->m_LogicalDevice, setLayout, nullptr);
            vkDestroyPipelineLayout(vulkanDevice->m_LogicalDevice, pipelineLayout, nullptr);
            vkDestroyPipeline(vulkanDevice->m_LogicalDevice, pipeline, nullptr);

        };
        VulkanDevice *vulkanDevice = nullptr;
        VkRender::RenderUtils *renderUtils = nullptr;

        struct Textures {
            TextureCubeMap environmentCube;
            Texture2D empty;
            Texture2D lutBrdf;
            TextureCubeMap irradianceCube;
            TextureCubeMap prefilteredCube;
        } textures;


        VkPipeline pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout setLayout;
        std::vector<VkDescriptorSet> descriptorSets{};
        VkPipelineLayout pipelineLayout{};
        VkDescriptorPool descriptorPool{};
        std::string selectedEnvironment = "papermill";

        struct ShaderValuesParams {
            glm::vec4 lightDir{};
            float exposure = 4.5f;
            float gamma = 2.2f;
            float prefilteredCubeMipLevels;
            float scaleIBLAmbient = 1.0f;
            float debugViewInputs = 0;
            float debugViewEquation = 0;
        } shaderValuesParams;

        VkRender::UBOMatrix uboMatrix;

        std::vector<Buffer> bufferSkyboxVert{};
        std::vector<Buffer> bufferSkyboxFrag{};

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex);

        void update();

        void generateCubemaps(const VkRender::GLTFModelComponent &component);

        void generateBRDFLUT();

        void setupUniformBuffers();

        void setupDescriptors(const VkRender::GLTFModelComponent &cube);

        void setupPipelines();

        VkPipelineShaderStageCreateInfo
        loadShader(std::string fileName, VkShaderStageFlagBits stage, VkShaderModule *module) {
            // Check if we have .spv extensions. If not then add it.
            std::size_t extension = fileName.find(".spv");
            if (extension == std::string::npos)
                fileName.append(".spv");
            Utils::loadShader((Utils::getShadersPath().append(fileName)).string().c_str(),
                              renderUtils->device->m_LogicalDevice, module);
            assert(module != VK_NULL_HANDLE);

            VkPipelineShaderStageCreateInfo shaderStage = {};
            shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStage.stage = stage;
            shaderStage.module = *module;
            shaderStage.pName = "main";
            Log::Logger::getInstance()->info("Loaded shader {} for stage {}", fileName, static_cast<uint32_t>(stage));
            return shaderStage;
        }

    };
};


#endif //MULTISENSE_VIEWER_SKYBOXGRAPHICSPIPELINECOMPONENT_H
