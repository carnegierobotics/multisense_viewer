//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_GUIMANAGER_H
#define MULTISENSE_GUIMANAGER_H


#include <type_traits>
#include <memory>
#include <vector>
#include <functional>
#include <glm/vec2.hpp>
#include <imgui/imgui_internal.h>

#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/Definitions.h"
#include "Viewer/ImGui/Layer.h"

namespace VkRender {
    class GuiManager {
    public:
        GuiObjectHandles handles{};
        GuiManager(VulkanDevice *vulkanDevice, const VkRenderPass& renderPass, const uint32_t& width, const uint32_t& height);
        ~GuiManager(){
            for (const auto& layerStack: m_LayerStack)
                layerStack->onDetach();
            vkDestroyPipeline(device->m_LogicalDevice, pipeline, nullptr);
            vkDestroyPipelineCache(device->m_LogicalDevice, pipelineCache, nullptr);
            vkDestroyPipelineLayout(device->m_LogicalDevice, pipelineLayout, nullptr);
            vkDestroyDescriptorPool(device->m_LogicalDevice, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(device->m_LogicalDevice, descriptorSetLayout, nullptr);
            for (auto * shaderModule: shaderModules) {
                vkDestroyShaderModule(device->m_LogicalDevice, shaderModule, nullptr);
            }

        };

        /**@brief Update function called from renderer. Function calls each layer in order to generate buffers for draw commands*/
        void update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height, const Input *pInput);
        /**@brief setup function called once vulkan renderer is setup. Function calls each layer in order to generate buffers for draw commands*/
        void setup(const uint32_t &width, const uint32_t &height, VkRenderPass const &renderPass);
        /**@brief Draw command called once per command buffer recording*/
        void drawFrame(VkCommandBuffer commandBuffer);
        /**@brief ReCreate buffers if they have changed in size*/
        bool updateBuffers();

        /** @brief Push a new GUI Layer to the layer stack. Layers are drawn in the order they are pushed. **/
        template<typename T>
        void pushLayer() {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type does not inherit Layer class!");
            m_LayerStack.emplace_back(std::make_shared<T>())->onAttach();
        }

    private:
        // UI params are set via push constants
        struct PushConstBlock {
            glm::vec2 scale;
            glm::vec2 translate;
        } pushConstBlock{};

        // GuiManager framework
        std::vector<std::shared_ptr<Layer>> m_LayerStack{};

        // Textures
        std::vector<Texture2D> iconTextures;
        std::vector<Texture2D> fontTexture;
        std::unique_ptr<Texture2D> gifTexture[99];

        // Vulkan resources for rendering the UI
        Buffer vertexBuffer;
        Buffer indexBuffer;
        int32_t vertexCount = 0;
        int32_t indexCount = 0;
        VkPipelineCache pipelineCache{};
        VkPipelineLayout pipelineLayout{};
        VkPipeline pipeline{};
        VkDescriptorPool descriptorPool{};
        VkDescriptorSetLayout descriptorSetLayout{};
        std::vector<VkDescriptorSet> fontDescriptors{};
        std::vector<VkDescriptorSet> imageIconDescriptors{};
        std::vector<VkDescriptorSet> gifImageDescriptors{};

        std::vector<VkShaderModule> shaderModules{};
        VulkanDevice *device = nullptr;

        // Initialization functions
        void initializeFonts();
        ImFont *loadFontFromFileName(std::string file, float fontSize);
        void loadImGuiTextureFromFileName(const std::string &file, uint32_t i);
        void loadAnimatedGif(const std::string &file);
    };
};
#endif //MULTISENSE_GUIMANAGER_H
