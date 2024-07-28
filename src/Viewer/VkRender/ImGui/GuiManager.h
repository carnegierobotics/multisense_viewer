/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/ImGui/GuiManager.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-4-19, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_GUIMANAGER_H
#define MULTISENSE_GUIMANAGER_H


#include <type_traits>
#include <memory>
#include <vector>
#include <glm/vec2.hpp>


#include <imgui_internal.h>

#include "Viewer/Tools/Utils.h"
#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/ImGui/Layers/LayerSupport/Layer.h"
#include "Viewer/VkRender/ImGui/Widgets.h"
#include "Viewer/VkRender/ImGui/Layers/LayerSupport/LayerFactory.h"

namespace VkRender {

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
        VkPipelineLayout pipelineLayout{};
        VkPipelineCache pipelineCache{};
        std::vector<VkShaderModule> shaderModules{};
        VulkanDevice *device = nullptr;

        GuiResources(VulkanDevice *d);
        GuiResources() = default;

        ImFont *font8{}, *font13{}, *font15, *font18{}, *font24{}, *fontIcons{};
        uint32_t fontCount = 0;
        uint32_t iconCount = 0;
        std::array<VkPipelineShaderStageCreateInfo, 2> shaders;
        ImFont *loadFontFromFileName(const std::filesystem::path& file, float fontSize, bool icons = false);
        ImFontAtlas fontAtlas;

        void loadAnimatedGif(const std::string &file);

        void loadImGuiTextureFromFileName(const std::string &file, uint32_t i);

        ~GuiResources(){
            vkDestroyPipelineCache(device->m_LogicalDevice, pipelineCache, nullptr);
            vkDestroyPipelineLayout(device->m_LogicalDevice, pipelineLayout, nullptr);
            vkDestroyDescriptorPool(device->m_LogicalDevice, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(device->m_LogicalDevice, descriptorSetLayout, nullptr);
            for (auto &shaderModule: shaderModules) {
                vkDestroyShaderModule(device->m_LogicalDevice, shaderModule, nullptr);
            }
        }
    };

    class GuiManager {
    public:
        GuiObjectHandles handles{};

        GuiManager(VulkanDevice *vulkanDevice, VkRenderPass const &renderPass, const uint32_t &width,
                   const uint32_t &height, VkSampleCountFlagBits msaaSamples, uint32_t imageCount, Renderer* ctx, ImGuiContext* imguiCtx, const GuiResources* guiResources); // TODO context should be pass by reference as it is no nullable?

        ~GuiManager() {
            //Log::Logger::getInstance()->info("Saving ImGui file: {}", (Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            //ImGui::SaveIniSettingsToDisk((Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            for (const auto &layerStack: m_LayerStack)
                layerStack->onDetach();

            vkDestroyPipeline(device->m_LogicalDevice, pipeline, nullptr);

            auto &userSetting = RendererConfig::getInstance().getUserSetting();
            userSetting.editorUiState.enableSecondaryView = handles.enableSecondaryView;
            userSetting.editorUiState.fixAspectRatio = handles.fixAspectRatio;

            ImGui::DestroyContext(m_imguiContext);
        };

        /**@brief Update function called from renderer. Function calls each layer in order to generate buffers for draw commands*/
        void update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height, const Input *pInput);

        /**@brief Draw command called once per command buffer recording*/
        void drawFrame(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t width, uint32_t height, uint32_t x,
                       uint32_t y);

        /**@brief ReCreate buffers if they have changed in size*/
        bool updateBuffers(uint32_t imageIndex);

        /** @brief Push a new GUI Layer to the layer stack. Layers are drawn in the order they are pushed. **/
        template<typename T>
        void pushLayer() {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type does not inherit Layer class!");
            m_LayerStack.emplace_back(std::make_shared<T>())->onAttach();
        }

        void pushLayer(const std::string& layerName) {
            auto layer = LayerFactory::createLayer(layerName);
            if(layer) {
                m_LayerStack.emplace_back(layer)->onAttach();
            } else {
                // Handle unknown layer case, e.g., throw an exception or log an error
            }
        }
        ImGuiContext* m_imguiContext;

    private:

        // GuiManager framework
        std::vector<std::shared_ptr<Layer>> m_LayerStack{};
        // Textures
        const GuiResources* m_guiResources;
        // Vulkan resources for rendering the UI
        //Buffer vertexBuffer;
        //Buffer indexBuffer;
        // Vulkan resources for rendering the UI
        std::vector<Buffer> vertexBuffer;
        std::vector<Buffer> indexBuffer;
        std::vector<int32_t> vertexCount{};
        std::vector<int32_t> indexCount{};

        VkPipeline pipeline{};
        GuiResources::PushConstBlock  pushConstBlock{};

        VulkanDevice *device = nullptr;
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> saveSettingsTimer;


    };
}
#endif //MULTISENSE_GUIMANAGER_H
