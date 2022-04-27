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
#include <MultiSense/src/tools/Populate.h>
#include <MultiSense/src/core/Buffer.h>
#include <MultiSense/src/tools/Utils.h>
#include <MultiSense/src/core/VulkanDevice.h>
#include "MultiSense/src/imgui/Layer.h"
#include <MultiSense/external/imgui/imgui.h>
#include <MultiSense/src/core/Texture.h>
#include "imgui_internal.h"

namespace ArEngine {


    class GuiManager {
    public:

        GuiObjectHandles handles;

        explicit GuiManager(VulkanDevice *vulkanDevice);
        ~GuiManager() = default;


        void update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height);

        void setup(float width, float height, VkRenderPass renderPass, VkQueue copyQueue, std::vector<VkPipelineShaderStageCreateInfo> *shaders);

        void drawFrame(VkCommandBuffer commandBuffer);
        bool updateBuffers();
        void setMenubarCallback(const std::function<void()> &menubarCallback) { m_MenubarCallback = menubarCallback; }

        template<typename T>
        void pushLayer() {
            static_assert(std::is_base_of<ArEngine::Layer, T>::value, "Pushed type does not inherit Layer class!");
            m_LayerStack.emplace_back(std::make_shared<T>())->OnAttach();
        }

    private:
        // UI params are set via push constants
        struct PushConstBlock {
            glm::vec2 scale;
            glm::vec2 translate;
        } pushConstBlock{};


        std::vector<std::shared_ptr<ArEngine::Layer>> m_LayerStack;
        std::function<void()> m_MenubarCallback;

        ImGuiIO *io{};
        ImVec4 clearColor;
        std::array<float, 50> frameTimes{};
        float frameTimeMin = 9999.0f, frameTimeMax = 0.0f;
        GuiLayerUpdateInfo updateInfo;

        Texture2D fontTexture;

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
        VkDescriptorSet descriptorSet{};

        VulkanDevice *device;

        void initializeFonts();
        ImFont *AddDefaultFont(float pixel_size);
        ImFont * loadFontFromFileName(std::string file, float fontSize);
    };

}
#endif //MULTISENSE_GUIMANAGER_H
