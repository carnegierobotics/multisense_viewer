//
// Created by magnus on 4/19/22.
//

//
// Adapted from Dear ImGui Vulkan example and from TheCherno's Walnut application
//


#ifdef WIN32
#define NOMINMAX
#endif

#include <stb_image.h>

#include "Viewer/ImGui/GuiManager.h"
#include "Viewer/ImGui/LayerExample.h"
#include "Viewer/ImGui/SideBar.h"
#include "Viewer/ImGui/DebugWindow.h"
#include "Viewer/ImGui/InteractionMenu.h"

namespace VkRender {

    const char* ImGuiGlfwGetClipboardText(void* user_data)
    {
        const char* str = glfwGetClipboardString((GLFWwindow*)user_data);
        return str;
    }

    void ImGuiGlfwSetClipboardText(void* user_data, const char* text)
    {
        glfwSetClipboardString((GLFWwindow*)user_data, text);
    }

    GuiManager::GuiManager(VulkanDevice *vulkanDevice, const VkRenderPass &renderPass, const uint32_t &width,
                           const uint32_t &height) {
        device = vulkanDevice;
        ImGui::CreateContext();

        handles.info = std::make_unique<GuiLayerUpdateInfo>();
        handles.info->deviceName = device->m_Properties.deviceName;
        handles.info->title = "GuiManager";
        ImGui::GetIO().GetClipboardTextFn = ImGuiGlfwGetClipboardText;
        ImGui::GetIO().SetClipboardTextFn = ImGuiGlfwSetClipboardText;

        initializeFonts();

        pushLayer<SideBar>();
        pushLayer<InteractionMenu>();
        pushLayer<LayerExample>();
        pushLayer<DebugWindow>();

        std::vector<VkPipelineShaderStageCreateInfo> shaders;

        // setup graphics pipeline
        setup(width, height, renderPass);

    }

    void
    GuiManager::update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height, const Input *pInput) {

        handles.info->frameTimer = frameTimer;
        handles.info->firstFrame = updateFrameGraph;
        handles.info->width = static_cast<float>(width);
        handles.info->height = static_cast<float>(height);
        handles.input = pInput;

        ImGui::NewFrame();

        {
            for (auto &layer: m_LayerStack) {
                layer->onUIRender(&handles);

            }
        }
        ImGui::Render();
        ImGui::EndFrame();


    }


// Update vertex and index buffer containing the imGui elements when required
    bool GuiManager::updateBuffers() {
        ImDrawData *imDrawData = ImGui::GetDrawData();


        bool updateCommandBuffers = false;
        // Note: Alignment is done inside buffer creation
        VkDeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
        VkDeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);

        if ((vertexBufferSize == 0) || (indexBufferSize == 0)) {
            return false;
        }

        // Update buffers only if vertex or index count has been changed compared to current buffer size

        // Vertex buffer
        if ((vertexBuffer.m_Buffer == VK_NULL_HANDLE) || (vertexCount != imDrawData->TotalVtxCount)) {
            vertexBuffer.unmap();
            vertexBuffer.destroy();
            if (VK_SUCCESS !=
                device->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     &vertexBuffer, vertexBufferSize))
                throw std::runtime_error("Failed to create vertex Buffer");
            vertexCount = imDrawData->TotalVtxCount;
            vertexBuffer.map();
            updateCommandBuffers = true;
        }

        // Index buffer
        if ((indexBuffer.m_Buffer == VK_NULL_HANDLE) || (indexCount < imDrawData->TotalIdxCount)) {
            indexBuffer.unmap();
            indexBuffer.destroy();
            if (VK_SUCCESS !=
                device->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     &indexBuffer, indexBufferSize))
                throw std::runtime_error("Failed to create index buffer");
            indexCount = imDrawData->TotalIdxCount;
            indexBuffer.map();
            updateCommandBuffers = true;
        }

        // Upload data
        ImDrawVert *vtxDst = (ImDrawVert *) vertexBuffer.mapped;
        ImDrawIdx *idxDst = (ImDrawIdx *) indexBuffer.mapped;

        for (int n = 0; n < imDrawData->CmdListsCount; n++) {
            const ImDrawList *cmd_list = imDrawData->CmdLists[n];
            memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
            vtxDst += cmd_list->VtxBuffer.Size;
            idxDst += cmd_list->IdxBuffer.Size;
        }

        // Flush to make writes visible to GPU
        vertexBuffer.flush();
        indexBuffer.flush();

        return updateCommandBuffers;
    }


    void GuiManager::drawFrame(VkCommandBuffer commandBuffer) {

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        //VkViewport viewport = Populate
        // ::viewport(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f);
        //vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        // UI scale and translate via push constants
        ImGuiIO *io = &ImGui::GetIO();
        pushConstBlock.scale = glm::vec2(2.0f / io->DisplaySize.x, 2.0f / io->DisplaySize.y);
        pushConstBlock.translate = glm::vec2(-1.0f);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstBlock),
                           &pushConstBlock);

        // Render commands
        ImDrawData *imDrawData = ImGui::GetDrawData();
        int32_t vertexOffset = 0;
        int32_t indexOffset = 0;

        if (imDrawData == nullptr) 
            return;

        if (imDrawData->CmdListsCount > 0) {

            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.m_Buffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer.m_Buffer, 0, VK_INDEX_TYPE_UINT16);

            for (int32_t i = 0; i < imDrawData->CmdListsCount; ++i) {
                const ImDrawList *cmd_list = imDrawData->CmdLists[i];
                 

                for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++) {
                    const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[j];

                    auto texture = (VkDescriptorSet) pcmd->GetTexID();
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                            &texture, 0, nullptr);

                    VkRect2D scissorRect{};
                    scissorRect.offset.x = std::max((int32_t) (pcmd->ClipRect.x), 0);
                    scissorRect.offset.y = std::max((int32_t) (pcmd->ClipRect.y), 0);
                    scissorRect.extent.width = (uint32_t) (pcmd->ClipRect.z - pcmd->ClipRect.x);
                    scissorRect.extent.height = (uint32_t) (pcmd->ClipRect.w - pcmd->ClipRect.y);
                    vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);
                    vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, pcmd->IdxOffset + indexOffset,
                                     pcmd->VtxOffset + vertexOffset, 0);
                }
                indexOffset += cmd_list->IdxBuffer.Size;
                vertexOffset += cmd_list->VtxBuffer.Size;
            }
        }
    }


    void GuiManager::setup(const uint32_t &width, const uint32_t &height, VkRenderPass const &renderPass) {
        VkShaderModule vtxModule{};
        Utils::loadShader((Utils::getShadersPath() + "Scene/imgui/ui.vert.spv").c_str(), device->m_LogicalDevice, &vtxModule);
        VkPipelineShaderStageCreateInfo vtxShaderStage = {};
        vtxShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vtxShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vtxShaderStage.module = vtxModule;
        vtxShaderStage.pName = "main";
        assert(vtxShaderStage.module != VK_NULL_HANDLE);
        shaderModules.push_back(vtxModule);

        VkShaderModule frgModule;
        Utils::loadShader((Utils::getShadersPath() + "Scene/imgui/ui.frag.spv").c_str(), device->m_LogicalDevice, &frgModule);
        VkPipelineShaderStageCreateInfo fragShaderStage = {};
        fragShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStage.module = frgModule;
        fragShaderStage.pName = "main";
        assert(fragShaderStage.module != VK_NULL_HANDLE);
        shaderModules.push_back(frgModule);

        std::array<VkPipelineShaderStageCreateInfo, 2> shaders{vtxShaderStage, fragShaderStage};


        ImGuiStyle &style = ImGui::GetStyle();
        style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
        style.Colors[ImGuiCol_PopupBg] = VkRender::Colors::CRLDarkGray425;
        style.Colors[ImGuiCol_WindowBg] = VkRender::Colors::CRLDarkGray425;
        style.Colors[ImGuiCol_Tab] = VkRender::Colors::CRLRed;
        style.Colors[ImGuiCol_TabActive] = VkRender::Colors::CRLRedActive;
        style.Colors[ImGuiCol_TabHovered] = VkRender::Colors::CRLRedHover;
        style.Colors[ImGuiCol_Button] = VkRender::Colors::CRLBlueIsh;

        style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.8f);


        // Dimensions
        ImGuiIO *io = &ImGui::GetIO();
        io->DisplaySize = ImVec2((float)width, (float)height);
        io->DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

        // Initialize all Vulkan resources used by the ui
        // Graphics pipeline

        io->ConfigFlags |= ImGuiConfigFlags_None;

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
            throw std::runtime_error("Failed to create pipeline layout");

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
                ::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);

        std::vector<VkDynamicState> dynamicStateEnables = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState =
                Populate
                ::pipelineDynamicStateCreateInfo(dynamicStateEnables);

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = Populate
        ::pipelineCreateInfo(pipelineLayout,
                             renderPass);


        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaders.size());
        pipelineCreateInfo.pStages = shaders.data();

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

        if (vkCreateGraphicsPipelines(device->m_LogicalDevice, pipelineCache, 1, &pipelineCreateInfo, nullptr,
                                      &pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");
    }

    void GuiManager::initializeFonts() {
        ImGuiIO *io = &ImGui::GetIO();
        io->Fonts->Clear();


        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
        setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };


        VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(),
                (uint32_t) setLayoutBindings.size());
        CHECK_RESULT(
                vkCreateDescriptorSetLayout(device->m_LogicalDevice, &layoutCreateInfo, nullptr,
                                            &descriptorSetLayout));


        uint32_t fontCount = 3, iconCount = 10, gifImageCount = 20;
        uint32_t setCount = fontCount + iconCount + gifImageCount;
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, setCount},

        };
        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, setCount);
        CHECK_RESULT(vkCreateDescriptorPool(device->m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));

        fontTexture.reserve(3);
        fontDescriptors.reserve(3);
        handles.info->font13 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 13); // TODO FIX PATHS
        handles.info->font18 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 18);
        handles.info->font24 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 24);
        io->Fonts->SetTexID(fontDescriptors[2]);

        // TODO use separate Descriptors to copy reference to imageButtonTextureDescriptors. Loss of memory alloc.
        imageIconDescriptors.resize(10);
        handles.info->imageButtonTextureDescriptor.resize(10);
        iconTextures.reserve(10);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_preview.png", 0);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_information.png", 1);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_configure.png", 2);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_auto_configure.png", 3);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_manual_configure.png", 4);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_playback.png", 5);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_single_layout.png", 6);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_double_layout.png", 7);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_quad_layout.png", 8);
        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_nine_layout.png", 9);


        loadAnimatedGif(Utils::getTexturePath() + "spinner.gif");

    }


    void GuiManager::loadAnimatedGif(const std::string &file) {
        int width = 0, height = 0, depth = 0, comp = 0;
        int *delays = nullptr;
        int channels = 4;

        std::ifstream input(file, std::ios::binary | std::ios::ate);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);

        stbi_uc *pixels = nullptr;
        std::vector<stbi_uc> buffer(size);
        if (input.read(reinterpret_cast<char *>(buffer.data()), size)) {
            pixels = stbi_load_gif_from_memory(buffer.data(),static_cast<int> (size), &delays, &width, &height, &depth, &comp, channels);
            if (!pixels)
                throw std::runtime_error("failed to load texture m_Image: " + file);
        }
        uint32_t imageSize = width * height * channels;

        handles.info->gif.width = width;
        handles.info->gif.height = height;
        handles.info->gif.totalFrames = depth;
        handles.info->gif.imageSize = imageSize;
        handles.info->gif.delay = (uint32_t *) delays;
        gifImageDescriptors.reserve((size_t) depth + 1);

        auto* pixelPointer = pixels; // Store original position in pixels

        for (int i = 0; i < depth; ++i) {
            VkDescriptorSet dSet{};
            gifTexture[i] = std::make_unique<Texture2D>(device);

            gifTexture[i]->fromBuffer(pixelPointer, handles.info->gif.imageSize, VK_FORMAT_R8G8B8A8_SRGB,
                                      handles.info->gif.width, handles.info->gif.height, device,
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
            write_desc[0].pImageInfo = &gifTexture[i]->m_Descriptor;
            vkUpdateDescriptorSets(device->m_LogicalDevice, 1, write_desc, 0, NULL);

            handles.info->gif.image[i] = reinterpret_cast<void *>(dSet);
            pixelPointer += handles.info->gif.imageSize;

            gifImageDescriptors.emplace_back(dSet);
        }
        stbi_image_free(pixels);
    }

    void GuiManager::loadImGuiTextureFromFileName(const std::string &file, uint32_t i) {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(file.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = (VkDeviceSize) texWidth * texHeight * texChannels;
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
            write_desc[0].pImageInfo = &iconTextures[i].m_Descriptor;
            vkUpdateDescriptorSets(device->m_LogicalDevice, 1, write_desc, 0, NULL);
        }

        handles.info->imageButtonTextureDescriptor[i] = reinterpret_cast<void *>(imageIconDescriptors[i]);
    }


    ImFont *GuiManager::loadFontFromFileName(std::string file, float fontSize) {
        ImFontConfig config;
        config.OversampleH = 2;
        config.OversampleV = 1;
        config.GlyphExtraSpacing.x = 1.0f;
        ImGuiIO *io = &ImGui::GetIO();
        ImFont *font = io->Fonts->AddFontFromFileTTF(file.c_str(), fontSize, &config);

        unsigned char *pixels;
        int width, height;
        io->Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        VkDeviceSize uploadSize = (VkDeviceSize) width * height * 4 * sizeof(char);

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
            write_desc[0].pImageInfo = &fontTexture.back().m_Descriptor;
            vkUpdateDescriptorSets(device->m_LogicalDevice, 1, write_desc, 0, NULL);
        }

        fontDescriptors.push_back(descriptor);
        return font;
    }

};