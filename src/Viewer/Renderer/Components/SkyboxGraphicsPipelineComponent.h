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
#include <random>

#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Renderer/Components/GLTFModelComponent.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Core/CommandBuffer.h"

namespace VkRender {


    struct SkyboxGraphicsPipelineComponent {

        SkyboxGraphicsPipelineComponent() = default;

        SkyboxGraphicsPipelineComponent(const SkyboxGraphicsPipelineComponent &) = default;

        SkyboxGraphicsPipelineComponent &operator=(const SkyboxGraphicsPipelineComponent &other) { return *this; }

        SkyboxGraphicsPipelineComponent(VkRender::RenderUtils *utils,
                                        const VkRender::GLTFModelComponent &modelComponent) {
            renderUtils = utils;
            vulkanDevice = utils->device;

            // Setup random number generation
            std::random_device rd;  // Obtain a random number from hardware
            std::mt19937 gen(rd()); // Seed the generator

            // List of texture file names
            std::vector<std::string> textureFiles = {
                    "kloppenheim.ktx2",
                    "skies.ktx2",
                    "snow_forest.ktx2",
            };
            std::uniform_int_distribution<> distr(0, textureFiles.size() - 1); // Define the range
            std::string selectedTexture = textureFiles[distr(gen)];

            textures.environmentCube.fromKtxFile(Utils::getTexturePath() / "Environments" / selectedTexture, vulkanDevice);

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

        VkRender::UBOMatrix uboMatrix;
        VkRender::ShaderValuesParams shaderValuesParams;

        std::vector<Buffer> bufferSkyboxVert{};
        std::vector<Buffer> bufferSkyboxFrag{};

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex);

        void update();

        void generateCubemaps(const VkRender::GLTFModelComponent &component);

        void generateBRDFLUT();

        void setupUniformBuffers();

        void setupDescriptors(const VkRender::GLTFModelComponent &cube);

        void setupPipelines();



    };
};


#endif //MULTISENSE_VIEWER_SKYBOXGRAPHICSPIPELINECOMPONENT_H
