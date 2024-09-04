//
// Created by magnus on 7/30/24.
//

#ifndef MULTISENSE_VIEWER_GUIRESOURCES_H
#define MULTISENSE_VIEWER_GUIRESOURCES_H

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include "Viewer/Application/pch.h"
#include "Viewer/VkRender/Core/Texture.h"

namespace VkRender {

    // TODO This should be replaced by a proper Asset manager
    // Should load elements from disk only when they are asked for
    // Should not have application specific strings in the constructor to identify resources
    // cherno youtube video about asset managers: https://youtu.be/38M-RwHG2hY?t=1373
    struct GuiResources {
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

        std::vector<Texture2D> iconTextures;
        std::vector<Texture2D> fontTexture;
        std::unique_ptr<Texture2D> gifTexture[99]; // Hold up to 99 frames
        VkDescriptorPool descriptorPool{};
        VkDescriptorSetLayout descriptorSetLayout{};
        std::vector<VkDescriptorSet> fontDescriptors{};
        std::vector<VkDescriptorSet> imageIconDescriptors{};
        std::vector<VkDescriptorSet> gifImageDescriptors{};
        std::vector<VkShaderModule> shaderModules{};
        VulkanDevice *device = nullptr;
        explicit GuiResources(VulkanDevice *d);
        GuiResources() = default;

        ImFont *font8{}, *font13{}, *font15, *font18{}, *font24{}, *fontIcons{};
        uint32_t fontCount = 0;
        uint32_t iconCount = 0;
        std::vector<VkPipelineShaderStageCreateInfo> shaders;
        ImFont *loadFontFromFileName(const std::filesystem::path& file, float fontSize, bool icons = false);
        ImFontAtlas fontAtlas;

        void loadAnimatedGif(const std::string &file);

        void loadImGuiTextureFromFileName(const std::string &file, uint32_t i);

        ~GuiResources(){
            vkDestroyDescriptorPool(device->m_LogicalDevice, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(device->m_LogicalDevice, descriptorSetLayout, nullptr);
            for (auto &shaderModule: shaderModules) {
                vkDestroyShaderModule(device->m_LogicalDevice, shaderModule, nullptr);
            }
        }
    };
}
#endif //MULTISENSE_VIEWER_GUIRESOURCES_H
