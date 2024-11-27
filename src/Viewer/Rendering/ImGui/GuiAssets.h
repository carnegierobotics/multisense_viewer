//
// Created by magnus on 7/30/24.
//

#ifndef MULTISENSE_VIEWER_GUIASSETS_H
#define MULTISENSE_VIEWER_GUIASSETS_H

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <vulkan/vulkan_core.h>

#include "Viewer/Application/pch.h"
#include "Viewer/Rendering/Core/VulkanDevice.h"
#include "Viewer/Rendering/Core/VulkanTexture.h"

namespace VkRender {
class Application;
    // TODO This should be replaced by a proper Asset manager
    // Should load elements from disk only when they are asked for
    // Should not have application specific strings in the constructor to identify resources
    // cherno youtube video about asset managers: https://youtu.be/38M-RwHG2hY?t=1373
    struct GuiAssets {
        // UI params are set via push constants
        struct PushConstBlock {
            glm::vec2 scale;
            glm::vec2 translate;
        } pushConstBlock{};

        /** @brief
        * Container to hold animated gif images
        */

        struct {
            ImTextureID image[20]{};
            uint32_t id{};
            uint32_t lastFrame = 0;
            uint32_t width{};
            uint32_t height{};
            uint32_t imageSize{};
            uint32_t totalFrames{};
            uint32_t *delay{};
            std::chrono::time_point<std::chrono::system_clock> lastUpdateTime = std::chrono::system_clock::now();
        } gif{};
        std::vector<std::shared_ptr<VulkanTexture2D>> fontTexture;
        std::vector<std::shared_ptr<VulkanTexture2D>> iconTextures;

        /*
        std::unique_ptr<Texture2D> gifTexture[99]; // Hold up to 99 frames
        */
        VkDescriptorPool descriptorPool{};
        VkDescriptorSetLayout descriptorSetLayout{};
        std::vector<VkDescriptorSet> fontDescriptors{};
        std::vector<VkDescriptorSet> imageIconDescriptors{};
        std::vector<VkDescriptorSet> gifImageDescriptors{};
        std::vector<VkShaderModule> shaderModules{};
        VulkanDevice& device;
        explicit GuiAssets(Application *context);

        ImFont *font8{}, *font13{}, *font15, *font18{}, *font24{}, *fontIcons{};
        uint32_t fontCount = 0;
        uint32_t iconCount = 0;
        std::vector<VkPipelineShaderStageCreateInfo> shaders;
        ImFontAtlas fontAtlas;

        void loadAnimatedGif(const std::string &file);

        ~GuiAssets(){
            vkDestroyDescriptorPool(device.m_LogicalDevice, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(device.m_LogicalDevice, descriptorSetLayout, nullptr);
            for (auto &shaderModule: shaderModules) {
                vkDestroyShaderModule(device.m_LogicalDevice, shaderModule, nullptr);
            }
        }

        ImFont *
        loadFontFromFileName(const std::filesystem::path &file, float fontSize, bool iconFont, Application *context);

        void loadImGuiTextureFromFileName(const std::string &file, uint32_t i, Application *context);
    };
}
#endif //MULTISENSE_VIEWER_GUIASSETS_H
