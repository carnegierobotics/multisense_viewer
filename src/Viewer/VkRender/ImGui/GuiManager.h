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


#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

#include "Viewer/Tools/Utils.h"
#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "Viewer/VkRender/Core/RenderDefinitions.h"
#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/Widgets.h"
#include "Viewer/VkRender/ImGui/LayerFactory.h"
#include "Viewer/VkRender/Editors/EditorIncludes.h"
#include "GuiResources.h"
#include "Viewer/VkRender/Core/VulkanGraphicsPipeline.h"

namespace VkRender {


    class GuiManager {
    public:
        GuiObjectHandles handles;

        GuiManager(VulkanDevice &vulkanDevice, VkRenderPass const &renderPass, EditorUI *editorUi,
                   VkSampleCountFlagBits msaaSamples, uint32_t imageCount, Renderer *ctx, ImGuiContext *imguiCtx,
                   const GuiResources *guiResources, SharedContextData* sharedData); // TODO context should be pass by reference as it is no nullable?

        ~GuiManager() {
            //Log::Logger::getInstance()->info("Saving ImGui file: {}", (Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            //ImGui::SaveIniSettingsToDisk((Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            for (const auto &layerStack: m_LayerStack)
                layerStack->onDetach();

            ImGui::DestroyContext(m_imguiContext);
        };

        void resize(uint32_t width, uint32_t height, VkRenderPass const &renderPass, VkSampleCountFlagBits msaaSamples, std::shared_ptr<GuiResources> guiResources);


        /**@brief Update function called from renderer. Function calls each layer in order to generate buffers for draw commands*/
        void update(bool updateFrameGraph, float frameTimer, EditorUI &editorUI, const Input *pInput);

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

        void pushLayer(const std::string &layerName);

        void setSceneContext(std::shared_ptr<Scene> scene){
            for (auto& layer : m_LayerStack){
                layer->setScene(scene);
            }
        }
        ImGuiContext *m_imguiContext = nullptr;

    private:

        // GuiManager framework
        std::vector<std::shared_ptr<Layer>> m_LayerStack{};
        GuiResources::PushConstBlock pushConstBlock{};
        // Textures
        const GuiResources *m_guiResources;
        // Vulkan resources for rendering the UI
        //Buffer vertexBuffer;
        //Buffer indexBuffer;
        // Vulkan resources for rendering the UI
        std::vector<Buffer> vertexBuffer;
        std::vector<Buffer> indexBuffer;
        std::vector<int32_t> vertexCount{};
        std::vector<int32_t> indexCount{};

        std::unique_ptr<VulkanGraphicsPipeline> m_pipeline;
        VulkanDevice &m_vulkanDevice;
        Renderer* m_context;
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> saveSettingsTimer;


        void createGraphicsPipeline(const VkRenderPass &renderPass, VkSampleCountFlagBits msaaSamples);

    };
}
#endif //MULTISENSE_GUIMANAGER_H
