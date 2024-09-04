/**
 * @file: MultiSense-Viewer/src/ImGui/GuiManager.cpp
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
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/
//
// Inspired by Dear ImGui Vulkan example and from TheCherno's Walnut application
// https://github.com/TheCherno/Walnut
// LICENSE: See MIT LICENSE in root folder of this repo


#include <stb_image.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <ranges>
#include "Viewer/VkRender/ImGui/GuiManager.h"
#include "Viewer/Application/Application.h"


namespace VkRender {

    const char *ImGuiGlfwGetClipboardText(void *userData) {
        const char *str = glfwGetClipboardString(reinterpret_cast<GLFWwindow *>(userData));
        return str;
    }

    void ImGuiGlfwSetClipboardText(void *userData, const char *text) {
        glfwSetClipboardString(reinterpret_cast<GLFWwindow *>(userData), text);
    }



    GuiManager::GuiManager(VulkanDevice &vulkanDevice, VkRenderPass const &renderPass, EditorUI *editorUi,
                           VkSampleCountFlagBits msaaSamples, uint32_t imageCount, Application* ctx,
                           ImGuiContext *imguiCtx,
                           const GuiResources *guiResources, SharedContextData* sharedData) : handles(sharedData),
            m_guiResources(guiResources), m_vulkanDevice(vulkanDevice), m_context(ctx) {

        vertexBuffer.resize(imageCount);
        indexBuffer.resize(imageCount);
        indexCount.resize(imageCount);
        vertexCount.resize(imageCount);
        m_imguiContext = imguiCtx;
        ImGui::SetCurrentContext(m_imguiContext);
        handles.info = std::make_unique<GuiLayerUpdateInfo>();

        handles.info->deviceName = m_vulkanDevice.m_Properties.deviceName;
        handles.info->title = "MultiSense Viewer";
        // Load UI info from file:
        handles.usageMonitor = ctx->m_usageMonitor;
        handles.editorUi = editorUi;

        ImGuiIO &io = ImGui::GetIO();
        io.GetClipboardTextFn = ImGuiGlfwGetClipboardText;
        io.SetClipboardTextFn = ImGuiGlfwSetClipboardText;
        io.DisplaySize = ImVec2(static_cast<float>(editorUi->width), static_cast<float>(editorUi->height));
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
        handles.info->imageButtonTextureDescriptor.resize(m_guiResources->iconCount);
        io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(m_guiResources->fontDescriptors[m_guiResources->fontCount]));

        ImGuiStyle &style = ImGui::GetStyle();
        style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
        style.Colors[ImGuiCol_PopupBg] = Colors::CRLDarkGray425;
        style.Colors[ImGuiCol_WindowBg] = Colors::CRLDarkGray425;
        style.Colors[ImGuiCol_FrameBg] = Colors::CRLFrameBG;
        style.Colors[ImGuiCol_Tab] = Colors::CRLRed;
        style.Colors[ImGuiCol_TabActive] = Colors::CRLRedActive;
        style.Colors[ImGuiCol_TabHovered] = Colors::CRLRedHover;
        style.Colors[ImGuiCol_Button] = Colors::CRLBlueIsh;
        style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.8f);


        handles.info->font8 = guiResources->font8;
        handles.info->font13 = guiResources->font13;
        handles.info->font15 = guiResources->font15;
        handles.info->font18 = guiResources->font18;
        handles.info->font24 = guiResources->font24;
        handles.info->fontIcons = guiResources->fontIcons;

        for (int i = 0; i < m_guiResources->iconCount; ++i) {
            handles.info->imageButtonTextureDescriptor[i] = reinterpret_cast<void *>(m_guiResources->imageIconDescriptors[i]);
        }
        for (int i = 0; i < m_guiResources->gif.totalFrames; ++i) {
            handles.info->gif.image[i] = m_guiResources->gifImageDescriptors[i];
        }

        VulkanGraphicsPipelineCreateInfo pipelineCreateInfo(renderPass, m_vulkanDevice);
        pipelineCreateInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        pipelineCreateInfo.msaaSamples = msaaSamples;
        pipelineCreateInfo.shaders = guiResources->shaders;
        pipelineCreateInfo.descriptorSetLayout = guiResources->descriptorSetLayout;
        pipelineCreateInfo.pushConstBlockSize = sizeof(GuiResources::PushConstBlock);
        pipelineCreateInfo.depthTesting = VK_FALSE;

        // Vertex bindings an attributes based on ImGui vertex definition
        std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
                Populate
                ::vertexInputBindingDescription(0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                Populate
                ::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert,
                                                                                          pos)),    // Location 0: Position
                Populate
                ::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT,
                                                  offsetof(ImDrawVert, uv)),    // Location 1: UV
                Populate
                ::vertexInputAttributeDescription(0, 2, VK_FORMAT_R8G8B8A8_UNORM,
                                                  offsetof(ImDrawVert, col)),    // Location 0: Color
        };
        VkPipelineVertexInputStateCreateInfo vertexInputState = Populate
        ::pipelineVertexInputStateCreateInfo();
        vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
        vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

        pipelineCreateInfo.vertexInputState = vertexInputState;
        m_pipeline = std::make_unique<VulkanGraphicsPipeline>(pipelineCreateInfo);

        // Create buffers to make sure they are initialized. We risk destroying without initializing if everything is null
    }

    void GuiManager::resize(uint32_t width, uint32_t height, const VkRenderPass &renderPass, VkSampleCountFlagBits msaaSamples, std::shared_ptr<GuiResources> guiResources) {
        ImGui::SetCurrentContext(m_imguiContext);
        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
        VulkanGraphicsPipelineCreateInfo pipelineCreateInfo(renderPass, m_vulkanDevice);
        pipelineCreateInfo.msaaSamples = msaaSamples;
        pipelineCreateInfo.shaders = guiResources->shaders;
        pipelineCreateInfo.descriptorSetLayout = guiResources->descriptorSetLayout;
        pipelineCreateInfo.pushConstBlockSize = sizeof(GuiResources::PushConstBlock);
        pipelineCreateInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                                                                         VK_FRONT_FACE_COUNTER_CLOCKWISE);
        pipelineCreateInfo.depthTesting = VK_FALSE;

        // Vertex bindings an attributes based on ImGui vertex definition
        std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
                Populate
                ::vertexInputBindingDescription(0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX),
        };
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                Populate
                ::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert,
                                                                                          pos)),    // Location 0: Position
                Populate
                ::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT,
                                                  offsetof(ImDrawVert, uv)),    // Location 1: UV
                Populate
                ::vertexInputAttributeDescription(0, 2, VK_FORMAT_R8G8B8A8_UNORM,
                                                  offsetof(ImDrawVert, col)),    // Location 0: Color
        };
        VkPipelineVertexInputStateCreateInfo vertexInputState = Populate
        ::pipelineVertexInputStateCreateInfo();
        vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
        vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

        pipelineCreateInfo.vertexInputState = vertexInputState;
        m_pipeline = std::make_unique<VulkanGraphicsPipeline>(pipelineCreateInfo);

    }

    void
    GuiManager::update(bool updateFrameGraph, float frameTimer, EditorUI &editorUI, const Input *pInput) {
        ImGui::SetCurrentContext(m_imguiContext);
        //Log::Logger::getInstance()->trace("Set ImGUI Context {} and updating", reinterpret_cast<uint64_t>(m_imguiContext));
        handles.editorUi = &editorUI;
        handles.info->frameTimer = frameTimer;
        handles.info->firstFrame = updateFrameGraph;
        handles.info->applicationWidth = static_cast<float>(editorUI.width);
        handles.info->applicationHeight = static_cast<float>(editorUI.height);

        handles.info->editorWidth = static_cast<float>(editorUI.width);
        handles.info->editorHeight = static_cast<float>(editorUI.height);
        handles.info->editorSize = ImVec2(handles.info->editorWidth, handles.info->editorHeight);
        handles.info->editorStartPos = ImVec2(0.0f, 0.0f);

        handles.info->aspect = static_cast<float>(editorUI.width) / static_cast<float>(editorUI.height);
        handles.input = pInput;

        ImGui::NewFrame();

        {
            for (auto &layer: m_LayerStack) {
                layer->onUIRender(handles);
            }
        }
        ImGui::Render();
        ImGui::EndFrame();

        // Save to config file every 5 seconds
        if (std::chrono::duration_cast<std::chrono::duration<float >>(
                std::chrono::steady_clock::now() - saveSettingsTimer).count() > 5.0f) {
            Log::Logger::getInstance()->traceWithFrequency("saveimgui", 12, "Saving ImGui file: {}",
                                                           (Utils::getSystemCachePath() /
                                                            "imgui.ini").string().c_str());
            ImGui::SaveIniSettingsToDisk((Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            saveSettingsTimer = std::chrono::steady_clock::now();
        }

    }


// Update vertex and index buffer containing the imGui elements when required
    bool GuiManager::updateBuffers(uint32_t currentFrame) {

        ImDrawData *imDrawData = ImGui::GetDrawData();

        // If we have no drawData return early
        if (!imDrawData)
            return false;

        bool updateCommandBuffers = false;
        // Note: Alignment is done inside buffer creation
        VkDeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
        VkDeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);

        if ((vertexBufferSize == 0) || (indexBufferSize == 0)) {
            return false;
        }

        // Update buffers only if vertex or index count has been changed compared to current buffer size

        // Vertex buffer
        if ((vertexBuffer[currentFrame].m_Buffer == VK_NULL_HANDLE) ||
            (vertexCount[currentFrame] != imDrawData->TotalVtxCount)) {
            vertexBuffer[currentFrame].unmap();
            vertexBuffer[currentFrame].destroy();
            if (VK_SUCCESS !=
                    m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                                &vertexBuffer[currentFrame], vertexBufferSize))
                throw std::runtime_error("Failed to create vertex Buffer");
            vertexCount[currentFrame] = imDrawData->TotalVtxCount;
            vertexBuffer[currentFrame].map();
            updateCommandBuffers = true;
        }

        // Index buffer
        if ((indexBuffer[currentFrame].m_Buffer == VK_NULL_HANDLE) ||
            (indexCount[currentFrame] < imDrawData->TotalIdxCount)) {
            indexBuffer[currentFrame].unmap();
            indexBuffer[currentFrame].destroy();
            if (VK_SUCCESS !=
                    m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                                &indexBuffer[currentFrame], indexBufferSize))
                throw std::runtime_error("Failed to create index buffer");
            indexCount[currentFrame] = imDrawData->TotalIdxCount;
            indexBuffer[currentFrame].map();
            updateCommandBuffers = true;
        }

        // Upload data
        auto *vtxDst = reinterpret_cast<ImDrawVert *>(vertexBuffer[currentFrame].mapped);
        auto *idxDst = reinterpret_cast<ImDrawIdx *>(indexBuffer[currentFrame].mapped);

        for (int n = 0; n < imDrawData->CmdListsCount; n++) {
            const ImDrawList *cmd_list = imDrawData->CmdLists[n];
            memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
            vtxDst += cmd_list->VtxBuffer.Size;
            idxDst += cmd_list->IdxBuffer.Size;
        }

        // Flush to make writes visible to GPU
        vertexBuffer[currentFrame].flush();
        indexBuffer[currentFrame].flush();

        return updateCommandBuffers;
    }


    void GuiManager::drawFrame(VkCommandBuffer commandBuffer, uint32_t currentFrame, uint32_t width, uint32_t height,
                               uint32_t x,
                               uint32_t y) {
        ImGui::SetCurrentContext(m_imguiContext);
        //Log::Logger::getInstance()->trace("Set ImGUI Context {} and drawing", reinterpret_cast<uint64_t>(m_imguiContext));

        // Need to update buffers
        updateBuffers(currentFrame);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->getPipeline());

        pushConstBlock.scale = glm::vec2(2.0f / static_cast<float>(width), 2.0f / static_cast<float>(height));
        pushConstBlock.translate = glm::vec2(-1.0f, -1.0f);
        vkCmdPushConstants(commandBuffer, m_pipeline->getPipelineLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(GuiResources::PushConstBlock),
                           &pushConstBlock);

        // Render commands
        ImDrawData *imDrawData = ImGui::GetDrawData();
        int32_t vertexOffset = 0;
        int32_t indexOffset = 0;

        if (imDrawData == nullptr)
            return;

        if (imDrawData->CmdListsCount > 0) {

            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer[currentFrame].m_Buffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer[currentFrame].m_Buffer, 0, VK_INDEX_TYPE_UINT16);

            for (int32_t i = 0; i < imDrawData->CmdListsCount; ++i) {
                const ImDrawList *cmd_list = imDrawData->CmdLists[i];


                for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++) {
                    const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[j];

                    auto texture = static_cast<VkDescriptorSet>(pcmd->GetTexID());
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            m_pipeline->getPipelineLayout(), 0, 1,
                                            &texture, 0, nullptr);

                    VkRect2D scissorRect{};
                    scissorRect.offset.x = std::max(static_cast<int32_t>(pcmd->ClipRect.x + x - 1),
                                                    0); // We're missing one pixel
                    scissorRect.offset.y = std::max(static_cast<int32_t>(pcmd->ClipRect.y + y - 1),
                                                    0); // We're missing one pixel
                    scissorRect.extent.width = static_cast<uint32_t>(pcmd->ClipRect.z - pcmd->ClipRect.x + 2); // We're missing one pixel
                    scissorRect.extent.height = static_cast<uint32_t>(pcmd->ClipRect.w - pcmd->ClipRect.y + 2); // We're missing one pixel

                    vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);
                    vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, pcmd->IdxOffset + indexOffset,
                                     pcmd->VtxOffset + vertexOffset, 0);
                }
                indexOffset += cmd_list->IdxBuffer.Size;
                vertexOffset += cmd_list->VtxBuffer.Size;
            }

            // Reset the scissors
            VkRect2D scissorRectFull;
            scissorRectFull.offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)};
            scissorRectFull.extent = {static_cast<uint32_t>(handles.info->applicationWidth),
                                      static_cast<uint32_t>(handles.info->applicationHeight)}; // Set these to your framebuffer or viewport dimensions
            //vkCmdSetScissor(commandBuffer, 0, 1, &scissorRectFull);
        }
    }

    void GuiManager::pushLayer(const std::string &layerName) {
            auto layer = LayerFactory::createLayer(layerName);
            layer->setContext(m_context);
            layer->setScene(m_context->activeScene());
            if (layer) {
                m_LayerStack.emplace_back(layer)->onAttach();
            } else {
                // Handle unknown layer case, e.g., throw an exception or log an error
            }

    }


}
