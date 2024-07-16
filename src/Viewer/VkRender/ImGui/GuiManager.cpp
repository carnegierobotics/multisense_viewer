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

#include "Viewer/VkRender/Renderer.h"

namespace VkRender {

    const char *ImGuiGlfwGetClipboardText(void *userData) {
        const char *str = glfwGetClipboardString(reinterpret_cast<GLFWwindow *>(userData));
        return str;
    }

    void ImGuiGlfwSetClipboardText(void *userData, const char *text) {
        glfwSetClipboardString(reinterpret_cast<GLFWwindow *>(userData), text);
    }

    GuiResources::GuiResources(VulkanDevice *d) : device(d) {

        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
        setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };

        uint32_t fonts = 5, icons = 10, gifImageCount = 20;
        uint32_t setCount = fonts + icons + gifImageCount;
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, setCount},

        };


        VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(),
                static_cast<uint32_t>(setLayoutBindings.size()));
        CHECK_RESULT(
                vkCreateDescriptorSetLayout(device->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                            &descriptorSetLayout));


        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, setCount);
        CHECK_RESULT(vkCreateDescriptorPool(device->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));


        /*

        if (std::filesystem::exists((Utils::getSystemCachePath() / "imgui.ini").string().c_str())) {
            Log::Logger::getInstance()->info("Loading imgui ini file from disk {}",
                                             (Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            ImGui::LoadIniSettingsFromDisk((Utils::getSystemCachePath() / "imgui.ini").string().c_str());
        } else {
            Log::Logger::getInstance()->info("ImGui ini file does not exist. {}",
                                             (Utils::getSystemCachePath() / "imgui.ini").string().c_str());
        }
        */



        fontTexture.reserve(fontCount);
        fontDescriptors.reserve(fontCount);
        font13 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 13);
        font8 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 8);
        font15 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 15);
        font18 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 18);
        font24 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 24);
        fontCount = fontDescriptors.size() - 1;



        auto iconFileNames = std::vector{
                "icon_preview.png",
                "icon_information.png",
                "icon_configure.png",
                "icon_auto_configure.png",
                "icon_manual_configure.png",
                "icon_playback.png",
                "icon_single_layout.png",
                "icon_double_layout.png",
                "icon_quad_layout.png",
                "icon_nine_layout.png"
        };
        // Reserve space for icon textures
        iconTextures.reserve(iconFileNames.size());
        iconCount = iconFileNames.size() - 1;
        imageIconDescriptors.resize(iconFileNames.size());
        // Base path for the texture files
        std::filesystem::path texturePath = Utils::getTexturePath();
        // Load textures using the filenames
        for (std::size_t index = 0; index < iconFileNames.size(); ++index) {
            const auto& filename = iconFileNames[index];
            loadImGuiTextureFromFileName((texturePath / filename).string(), index);


        }
        loadAnimatedGif(Utils::getTexturePath().append("spinner.gif").string());

        // setup graphics pipeline
        VkShaderModule vtxModule{};
        Utils::loadShader((Utils::getShadersPath().append("Scene/imgui/ui.vert.spv")).string().c_str(),
                          device->m_LogicalDevice, &vtxModule);
        VkPipelineShaderStageCreateInfo vtxShaderStage = {};
        vtxShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vtxShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vtxShaderStage.module = vtxModule;
        vtxShaderStage.pName = "main";
        assert(vtxShaderStage.module != VK_NULL_HANDLE);
        shaderModules.push_back(vtxModule);

        VkShaderModule frgModule;
        Utils::loadShader((Utils::getShadersPath().append("Scene/imgui/ui.frag.spv")).string().c_str(),
                          device->m_LogicalDevice, &frgModule);
        VkPipelineShaderStageCreateInfo fragShaderStage = {};
        fragShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStage.module = frgModule;
        fragShaderStage.pName = "main";
        assert(fragShaderStage.module != VK_NULL_HANDLE);
        shaderModules.push_back(frgModule);

        shaders = std::array<VkPipelineShaderStageCreateInfo, 2> {vtxShaderStage, fragShaderStage};


        // Pipeline cache
        VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
        pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(device->m_LogicalDevice, &pipelineCacheCreateInfo, nullptr, &pipelineCache) !=
            VK_SUCCESS)
            throw std::runtime_error("Failed to create Pipeline Cache");

        // Pipeline layout
        // Push constants for UI rendering parameters
        VkPushConstantRange pushConstantRange = Populate::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT,
                                                                            sizeof(PushConstBlock), 0);
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = Populate::pipelineLayoutCreateInfo(
                &descriptorSetLayout, 1);
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        if (
                vkCreatePipelineLayout(device->m_LogicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) !=
                VK_SUCCESS)
            throw std::runtime_error("Failed to create m_Pipeline layout");


    }

    GuiManager::GuiManager(VulkanDevice *vulkanDevice, VkRenderPass const &renderPass, const uint32_t &width,
                           const uint32_t &height, VkSampleCountFlagBits msaaSamples, uint32_t imageCount, Renderer* ctx, ImGuiContext* imguiCtx, const GuiResources* guiResources) : m_guiResources(guiResources) {
        device = vulkanDevice;
        vertexBuffer.resize(imageCount);
        indexBuffer.resize(imageCount);
        indexCount.resize(imageCount);
        vertexCount.resize(imageCount);
        m_imguiContext = imguiCtx;
        ImGui::SetCurrentContext(m_imguiContext);
        handles.info = std::make_unique<GuiLayerUpdateInfo>();
        handles.multiSenseRendererBridge = std::make_unique<MultiSense::MultiSenseRendererBridge>();
        handles.multiSenseRendererGigEVisionBridge = std::make_unique<MultiSense::MultiSenseRendererGigEVisionBridge>();
        handles.info->deviceName = device->m_Properties.deviceName;
        handles.info->title = "MultiSense Viewer";
        // Load UI info from file:
        auto &userSetting = RendererConfig::getInstance().getUserSetting();
        handles.enableSecondaryView = userSetting.editorUiState.enableSecondaryView;
        handles.fixAspectRatio = userSetting.editorUiState.fixAspectRatio;
        handles.m_context = ctx;
        handles.usageMonitor = ctx->m_usageMonitor;

        ImGuiIO& io = ImGui::GetIO();
        io.GetClipboardTextFn = ImGuiGlfwGetClipboardText;
        io.SetClipboardTextFn = ImGuiGlfwSetClipboardText;
        io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));
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

        for (int i = 0; i < m_guiResources->iconCount; ++i){
            handles.info->imageButtonTextureDescriptor[i] = reinterpret_cast<void *>(m_guiResources->imageIconDescriptors[i]);
        }
        for (int i = 0; i < m_guiResources->gif.totalFrames; ++i) {
            handles.info->gif.image[i] = m_guiResources->gifImageDescriptors[i];
        }

        Log::Logger::getInstance()->info("Set ImGUI Context {} displaySize to: {}x{}", reinterpret_cast<uint64_t>(m_imguiContext), width, height);

        // Setup graphics pipeline for UI rendering
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
                Populate::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0,
                                                               VK_FALSE);

        VkPipelineRasterizationStateCreateInfo rasterizationState =
                Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                               VK_FRONT_FACE_COUNTER_CLOCKWISE);

        // Enable blending
        VkPipelineColorBlendAttachmentState blendAttachmentState{};
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlendState =
                Populate::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        VkPipelineDepthStencilStateCreateInfo depthStencilState =
                Populate
                ::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL);

        VkPipelineViewportStateCreateInfo viewportState =
                Populate
                ::pipelineViewportStateCreateInfo(1, 1, 0);

        VkPipelineMultisampleStateCreateInfo multisampleState =
                Populate
                ::pipelineMultisampleStateCreateInfo(msaaSamples);

        std::vector<VkDynamicState> dynamicStateEnables = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState =
                Populate
                ::pipelineDynamicStateCreateInfo(dynamicStateEnables);

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = Populate
        ::pipelineCreateInfo(m_guiResources->pipelineLayout,
                             renderPass);


        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(m_guiResources->shaders.size());
        pipelineCreateInfo.pStages = m_guiResources->shaders.data();

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

        pipelineCreateInfo.pVertexInputState = &vertexInputState;

        if (vkCreateGraphicsPipelines(device->m_LogicalDevice, m_guiResources->pipelineCache, 1, &pipelineCreateInfo, nullptr,
                                      &pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics m_Pipeline");
    }

    void
    GuiManager::update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height, const Input *pInput) {
        ImGui::SetCurrentContext(m_imguiContext);
        Log::Logger::getInstance()->trace("Set ImGUI Context {} and updating", reinterpret_cast<uint64_t>(m_imguiContext));

        handles.info->frameTimer = frameTimer;
        handles.info->firstFrame = updateFrameGraph;
        handles.info->width = static_cast<float>(width);
        handles.info->height = static_cast<float>(height);
        handles.info->aspect = static_cast<float>(width) / static_cast<float>(height);
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
                                              (Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            ImGui::SaveIniSettingsToDisk((Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            saveSettingsTimer = std::chrono::steady_clock::now();
        }

        handles.multiSenseRendererBridge->update(); // TODO reconsider if we should call crl updates here?
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
                device->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
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
                device->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
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
        Log::Logger::getInstance()->trace("Set ImGUI Context {} and drawing", reinterpret_cast<uint64_t>(m_imguiContext));

        // Need to update buffers
        updateBuffers(currentFrame);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        pushConstBlock.scale = glm::vec2(2.0f / static_cast<float>(width), 2.0f /  static_cast<float>(height));
        pushConstBlock.translate = glm::vec2(-1.0f, -1.0f);
        vkCmdPushConstants(commandBuffer, m_guiResources->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GuiResources::PushConstBlock),
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
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_guiResources->pipelineLayout, 0, 1,
                                            &texture, 0, nullptr);

                    VkRect2D scissorRect{};
                    scissorRect.offset.x = std::max(static_cast<int32_t>(pcmd->ClipRect.x + x), 0);
                    scissorRect.offset.y = std::max(static_cast<int32_t>(pcmd->ClipRect.y + y), 0);
                    scissorRect.extent.width = static_cast<uint32_t>(pcmd->ClipRect.z - pcmd->ClipRect.x);
                    scissorRect.extent.height = static_cast<uint32_t>(pcmd->ClipRect.w - pcmd->ClipRect.y);
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
            scissorRectFull.extent = {static_cast<uint32_t>(handles.info->width),
                                      static_cast<uint32_t>(handles.info->height)}; // Set these to your framebuffer or viewport dimensions
            //vkCmdSetScissor(commandBuffer, 0, 1, &scissorRectFull);
        }
    }

    void GuiResources::loadImGuiTextureFromFileName(const std::string &file, uint32_t i) {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(file.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * texChannels);
        if (!pixels) {
            throw std::runtime_error("failed to load texture m_Image: " + file);
        }

        iconTextures.emplace_back(pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, static_cast<uint32_t>(texWidth),
                                  static_cast<uint32_t>(texHeight), device,
                                  device->m_TransferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            CHECK_RESULT(vkAllocateDescriptorSets(device->m_LogicalDevice, &alloc_info, &imageIconDescriptors[i]));
        }
        // Update the Descriptor Set:
        {

            VkWriteDescriptorSet write_desc[1] = {};
            write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_desc[0].dstSet = imageIconDescriptors[i];
            write_desc[0].descriptorCount = 1;
            write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_desc[0].pImageInfo = &iconTextures[i].m_descriptor;
            vkUpdateDescriptorSets(device->m_LogicalDevice, 1, write_desc, 0, NULL);
        }

    }


    ImFont *GuiResources::loadFontFromFileName(std::string file, float fontSize) {
        ImFontConfig config;
        config.OversampleH = 2;
        config.OversampleV = 1;
        config.GlyphExtraSpacing.x = 1.0f;
        ImFont *font = fontAtlas.AddFontFromFileTTF(file.c_str(), fontSize, &config);

        unsigned char *pixels;
        int width, height;
        fontAtlas.GetTexDataAsRGBA32(&pixels, &width, &height);
        auto uploadSize = width * height * 4 * sizeof(char);

        fontTexture.emplace_back(pixels, uploadSize,
                                 VK_FORMAT_R8G8B8A8_UNORM,
                                 width, height, device,
                                 device->m_TransferQueue);
        VkDescriptorSet descriptor{};
        // descriptors
        // Create Descriptor Set:
        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            VkResult res = vkAllocateDescriptorSets(device->m_LogicalDevice, &alloc_info, &descriptor);
            if (res != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate descriptorset");
            }
        }
        // Update the Descriptor Set:
        {
            VkWriteDescriptorSet write_desc[1] = {};
            write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_desc[0].dstSet = descriptor;
            write_desc[0].descriptorCount = 1;
            write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_desc[0].pImageInfo = &fontTexture.back().m_descriptor;
            vkUpdateDescriptorSets(device->m_LogicalDevice, 1, write_desc, 0, nullptr);
        }

        fontDescriptors.push_back(descriptor);
        return font;
    }


    void GuiResources::loadAnimatedGif(const std::string &file) {
        int width = 0, height = 0, depth = 0, comp = 0;
        int *delays = nullptr;
        int channels = 4;

        std::ifstream input(file, std::ios::binary | std::ios::ate);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);

        stbi_uc *pixels = nullptr;
        std::vector<stbi_uc> buffer(size);
        if (input.read(reinterpret_cast<char *>(buffer.data()), size)) {
            pixels = stbi_load_gif_from_memory(buffer.data(), static_cast<int> (size), &delays, &width, &height, &depth,
                                               &comp, channels);
            if (!pixels)
                throw std::runtime_error("failed to load texture m_Image: " + file);
        }
        uint32_t imageSize = width * height * channels;

        gif.width = width;
        gif.height = height;
        gif.totalFrames = depth;
        gif.imageSize = imageSize;
        gif.delay = reinterpret_cast<uint32_t *>( delays);
        gifImageDescriptors.reserve(static_cast<size_t>(depth) + 1);

        auto *pixelPointer = pixels; // Store original position in pixels

        for (int i = 0; i < depth; ++i) {
            VkDescriptorSet dSet{};
            gifTexture[i] = std::make_unique<Texture2D>(pixelPointer, imageSize, VK_FORMAT_R8G8B8A8_SRGB,
                                      width, height, device,
                                      device->m_TransferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            // Create Descriptor Set:

            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            CHECK_RESULT(vkAllocateDescriptorSets(device->m_LogicalDevice, &alloc_info, &dSet));


            // Update the Descriptor Set:
            VkWriteDescriptorSet write_desc[1] = {};
            write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_desc[0].dstSet = dSet;
            write_desc[0].descriptorCount = 1;
            write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_desc[0].pImageInfo = &gifTexture[i]->m_descriptor;
            vkUpdateDescriptorSets(device->m_LogicalDevice, 1, write_desc, 0, NULL);
            pixelPointer += imageSize;

            gifImageDescriptors.emplace_back(dSet);
        }
        stbi_image_free(pixels);
    }




}
