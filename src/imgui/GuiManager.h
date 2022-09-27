//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_GUIMANAGER_H
#define MULTISENSE_GUIMANAGER_H


#include <type_traits>
#include <memory>
#include <vector>
#include <functional>
#include "glm/vec2.hpp"
#include <MultiSense/src/Tools/Populate.h>
#include <MultiSense/src/Core/Buffer.h>
#include <MultiSense/src/Tools/Utils.h>
#include <MultiSense/src/Core/VulkanDevice.h>
#include "MultiSense/src/imgui/Layer.h"
#include <MultiSense/src/Core/Texture.h>
#include "imgui_internal.h"

namespace MultiSense {


    class GuiManager {
    public:

        GuiObjectHandles handles{};

        explicit GuiManager(VulkanDevice *vulkanDevice);

        ~GuiManager() = default;


        void update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height, const Input *pInput);

        void setup(float width, float height, VkRenderPass renderPass, VkQueue copyQueue,
                   std::vector<VkPipelineShaderStageCreateInfo> *shaders);

        void drawFrame(VkCommandBuffer commandBuffer);

        bool updateBuffers();

        void setMenubarCallback(const std::function<void()> &menubarCallback) { m_MenubarCallback = menubarCallback; }

        template<typename T>
        void pushLayer() {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type does not inherit Layer class!");
            m_LayerStack.emplace_back(std::make_shared<T>())->OnAttach();
        }

    private:
        // UI params are set via push constants
        struct PushConstBlock {
            glm::vec2 scale;
            glm::vec2 translate;
        } pushConstBlock{};


        std::vector<std::shared_ptr<Layer>> m_LayerStack;
        std::function<void()> m_MenubarCallback;

        ImGuiIO *io{};
        Texture2D fontTexture{};
        Texture2D iconTexture{};
        Texture2D gifTexture[99];

        // Vulkan resources for rendering the UI
        VkSampler sampler{};
        Buffer vertexBuffer;
        Buffer indexBuffer;
        int32_t vertexCount = 0;
        int32_t indexCount = 0;
        VkPipelineCache pipelineCache{};
        VkPipelineLayout pipelineLayout{};
        VkPipeline pipeline{};
        VkDescriptorPool descriptorPool{};
        VkDescriptorSetLayout descriptorSetLayout{};
        VkDescriptorSet fontDescriptor{};
        VkDescriptorSet imageIconDescriptor{};

        VkDescriptorSet gifImageDescriptors[20];     // TODO crude and "quick" implementation. Lots of missed memory and uses more memory than necessary. Fix in the future


        VulkanDevice *device;

        void initializeFonts();

        ImFont *AddDefaultFont(float pixel_size);

        ImFont *loadFontFromFileName(std::string file, float fontSize);

        void loadImGuiTextureFromFileName(const std::string& file);

        void loadAnimatedGif(const std::string &file);

        void loadNextGifFrame();
    };
};
#endif //MULTISENSE_GUIMANAGER_H
