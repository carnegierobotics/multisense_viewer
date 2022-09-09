//
// Created by magnus on 4/19/22.
//

//
// Adapted from Dear ImGui Vulkan example and from TheCherno's Walnut application
//


#ifdef WIN32
#define NOMINMAX
#endif


#include <MultiSense/src/tools/Macros.h>
#include "GuiManager.h"
#include "stb_image.h"

namespace AR {
    GuiManager::GuiManager(VulkanDevice *vulkanDevice) {
        device = vulkanDevice;
        ImGui::CreateContext();

        handles.info = new GuiLayerUpdateInfo();
        handles.info->deviceName = device->properties.deviceName;
        handles.info->title = "GuiManager";
        io = &ImGui::GetIO();

        initializeFonts();

    }

    void GuiManager::update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height) {

        handles.info->frameTimer = frameTimer;
        handles.info->firstFrame = updateFrameGraph;
        handles.info->width = width;
        handles.info->height = height;

        ImGui::NewFrame();

        {
            for (auto &layer: m_LayerStack) {
                layer->OnUIRender(&handles);

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
        if ((vertexBuffer.buffer == VK_NULL_HANDLE) || (vertexCount != imDrawData->TotalVtxCount)) {
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
        if ((indexBuffer.buffer == VK_NULL_HANDLE) || (indexCount < imDrawData->TotalIdxCount)) {
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
        pushConstBlock.scale = glm::vec2(2.0f / io->DisplaySize.x, 2.0f / io->DisplaySize.y);
        pushConstBlock.translate = glm::vec2(-1.0f);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstBlock),
                           &pushConstBlock);

        // Render commands
        ImDrawData *imDrawData = ImGui::GetDrawData();
        int32_t vertexOffset = 0;
        int32_t indexOffset = 0;

        if (imDrawData == nullptr) return;

        if (imDrawData->CmdListsCount > 0) {

            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

            for (int32_t i = 0; i < imDrawData->CmdListsCount; i++) {
                const ImDrawList *cmd_list = imDrawData->CmdLists[i];


                for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++) {
                    const ImDrawCmd *pcmd = &cmd_list->CmdBuffer[j];

                    auto texture = (VkDescriptorSet) pcmd->TextureId;
                    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                            &texture, 0, nullptr);

                    VkRect2D scissorRect;
                    scissorRect.offset.x = std::max((int32_t) (pcmd->ClipRect.x), 0);
                    scissorRect.offset.y = std::max((int32_t) (pcmd->ClipRect.y), 0);
                    scissorRect.extent.width = (uint32_t) (pcmd->ClipRect.z - pcmd->ClipRect.x);
                    scissorRect.extent.height = (uint32_t) (pcmd->ClipRect.w - pcmd->ClipRect.y);
                    vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);
                    vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
                    indexOffset += pcmd->ElemCount;
                }
                vertexOffset += cmd_list->VtxBuffer.Size;
            }
        }
    }


    void GuiManager::setup(float width, float height, VkRenderPass renderPass, VkQueue copyQueue,
                           std::vector<VkPipelineShaderStageCreateInfo> *shaders) {
        ImGuiStyle &style = ImGui::GetStyle();
        style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
        // Dimensions
        io = &ImGui::GetIO();
        io->DisplaySize = ImVec2(width, height);
        io->DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

        // Initialize all Vulkan resources used by the ui
        // Graphics pipeline

        io->ConfigFlags |= ImGuiConfigFlags_None;

        // Pipeline cache
        VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
        pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(device->logicalDevice, &pipelineCacheCreateInfo, nullptr, &pipelineCache) !=
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
                vkCreatePipelineLayout(device->logicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) !=
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


        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaders->size());
        pipelineCreateInfo.pStages = shaders->data();

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

        if (vkCreateGraphicsPipelines(device->logicalDevice, pipelineCache, 1, &pipelineCreateInfo, nullptr,
                                      &pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");
    }

    void GuiManager::initializeFonts() {
        io->Fonts->Clear();

        handles.info->font13 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 13); // TODO FIX PATHS
        handles.info->font18 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 18);
        handles.info->font24 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 24);
        io->Fonts->SetTexID(reinterpret_cast<void *>(fontDescriptor));


        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_preview.png");
        handles.info->imageButtonTextureDescriptor[0] = reinterpret_cast<void *>(imageIconDescriptor);

        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_information.png");
        handles.info->imageButtonTextureDescriptor[1] = reinterpret_cast<void *>(imageIconDescriptor);

        loadImGuiTextureFromFileName(Utils::getTexturePath() + "icon_configure.png");
        handles.info->imageButtonTextureDescriptor[2] = reinterpret_cast<void *>(imageIconDescriptor);

        loadAnimatedGif(Utils::getTexturePath() + "spinner.gif");

    }


    // TODO crude and "quick" implementation. Lots of missed memory and uses way more memory than necessary. Fix in the future
    void GuiManager::loadAnimatedGif(const std::string &file) {
        int width, height, depth, comp;
        int *delays;
        int channels = 4;

        std::ifstream input(file, std::ios::binary | std::ios::ate);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);

        stbi_uc *pixels;
        std::vector<stbi_uc> buffer(size);
        if (input.read(reinterpret_cast<char *>(buffer.data()), size)) {
            pixels = stbi_load_gif_from_memory(buffer.data(), size, &delays, &width, &height, &depth, &comp, channels);
            if (!pixels)
                throw std::runtime_error("failed to load texture image: " + file);
        }
        VkDeviceSize imageSize = width * height * channels;

        handles.info->gif.pixels = (unsigned char *) malloc(imageSize * depth);
        memcpy(handles.info->gif.pixels, pixels, imageSize * depth);

        handles.info->gif.width = width;
        handles.info->gif.height = height;
        handles.info->gif.totalFrames = depth;
        handles.info->gif.imageSize = imageSize;
        handles.info->gif.delay = (uint32_t *)delays;


        for (int i = 0; i < depth; ++i) {

            gifTexture[i].fromBuffer(handles.info->gif.pixels, handles.info->gif.imageSize, VK_FORMAT_R8G8B8A8_SRGB,
                                     handles.info->gif.width, handles.info->gif.height, device,
                                     device->transferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


            // Descriptor Layout

            {
                std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
                setLayoutBindings = {
                        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                };


                VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                        setLayoutBindings.data(),
                        setLayoutBindings.size());

                CHECK_RESULT(
                        vkCreateDescriptorSetLayout(device->logicalDevice, &layoutCreateInfo, nullptr,
                                                    &descriptorSetLayout));
            }

            // Descriptor Pool
            {
                uint32_t imageDescriptorSamplerCount = (3 * 5);
                std::vector<VkDescriptorPoolSize> poolSizes = {
                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

                };
                VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, 5);
                CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &poolCreateInfo, nullptr, &descriptorPool));


            }

            // descriptors

            // Create Descriptor Set:
            {
                VkDescriptorSetAllocateInfo alloc_info = {};
                alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                alloc_info.descriptorPool = descriptorPool;
                alloc_info.descriptorSetCount = 1;
                alloc_info.pSetLayouts = &descriptorSetLayout;
                CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice, &alloc_info, &gifImageDescriptors[i]));
            }

            // Update the Descriptor Set:
            {

                VkWriteDescriptorSet write_desc[1] = {};
                write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_desc[0].dstSet = gifImageDescriptors[i];
                write_desc[0].descriptorCount = 1;
                write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_desc[0].pImageInfo = &gifTexture[i].descriptor;
                vkUpdateDescriptorSets(device->logicalDevice, 1, write_desc, 0, NULL);
            }

            handles.info->gif.image[i] = reinterpret_cast<void *>(gifImageDescriptors[i]);
            handles.info->gif.pixels += handles.info->gif.imageSize;
        }

    }

    ArEngine::ImageElement GuiManager::loadImGuiTextureFromFileName(const std::string &file) {


        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(file.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        if (!pixels) {
            throw std::runtime_error("failed to load texture image: " + file);
        }

        iconTexture.fromBuffer(pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, device,
                               device->transferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);




        // Descriptor Layout

        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };


            VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                    setLayoutBindings.data(),
                    setLayoutBindings.size());

            CHECK_RESULT(
                    vkCreateDescriptorSetLayout(device->logicalDevice, &layoutCreateInfo, nullptr,
                                                &descriptorSetLayout));
        }

        // Descriptor Pool
        {
            uint32_t imageDescriptorSamplerCount = (3 * 5);
            std::vector<VkDescriptorPoolSize> poolSizes = {
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

            };
            VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, 5);
            CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &poolCreateInfo, nullptr, &descriptorPool));


        }

        // descriptors

        // Create Descriptor Set:
        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice, &alloc_info, &imageIconDescriptor));
        }

        // Update the Descriptor Set:
        {

            VkWriteDescriptorSet write_desc[1] = {};
            write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_desc[0].dstSet = imageIconDescriptor;
            write_desc[0].descriptorCount = 1;
            write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_desc[0].pImageInfo = &iconTexture.descriptor;
            vkUpdateDescriptorSets(device->logicalDevice, 1, write_desc, 0, NULL);
        }


        return {imageIconDescriptor, texWidth, texHeight};
    }


    ImFont *GuiManager::loadFontFromFileName(std::string file, float fontSize) {
        ImFont *font;
        ImFontConfig config;
        config.OversampleH = 2;
        config.OversampleV = 1;
        config.GlyphExtraSpacing.x = 1.0f;
        font = io->Fonts->AddFontFromFileTTF(file.c_str(), fontSize, &config);

        unsigned char *pixels;
        int width, height;
        io->Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        size_t uploadSize = width * height * 4 * sizeof(char);

        {
            fontTexture.fromBuffer(pixels, uploadSize,
                                   VK_FORMAT_R8G8B8A8_UNORM,
                                   width, height, device,
                                   device->transferQueue);

        }

        // Descriptor Layout

        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
            setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };


            VkDescriptorSetLayoutCreateInfo layoutCreateInfo = Populate::descriptorSetLayoutCreateInfo(
                    setLayoutBindings.data(),
                    setLayoutBindings.size());
            CHECK_RESULT(
                    vkCreateDescriptorSetLayout(device->logicalDevice, &layoutCreateInfo, nullptr,
                                                &descriptorSetLayout));
        }

        // Descriptor Pool
        {
            uint32_t imageDescriptorSamplerCount = (3 * 5);
            std::vector<VkDescriptorPoolSize> poolSizes = {
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageDescriptorSamplerCount},

            };
            VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, 5);
            CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &poolCreateInfo, nullptr, &descriptorPool));


        }

        // descriptors
        {
            // Create Descriptor Set:
            {
                VkDescriptorSetAllocateInfo alloc_info = {};
                alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                alloc_info.descriptorPool = descriptorPool;
                alloc_info.descriptorSetCount = 1;
                alloc_info.pSetLayouts = &descriptorSetLayout;
                CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice, &alloc_info, &fontDescriptor));
            }

            // Update the Descriptor Set:
            {

                VkWriteDescriptorSet write_desc[1] = {};
                write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_desc[0].dstSet = fontDescriptor;
                write_desc[0].descriptorCount = 1;
                write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_desc[0].pImageInfo = &fontTexture.descriptor;
                vkUpdateDescriptorSets(device->logicalDevice, 1, write_desc, 0, NULL);
            }
        }


        return font;
    }

    ImFont *GuiManager::AddDefaultFont(float pixel_size) {
        ImGuiIO &io = ImGui::GetIO();
        ImFontConfig config;
        config.SizePixels = pixel_size;
        config.OversampleH = 2;
        config.OversampleV = 1;
        config.FontDataOwnedByAtlas = false;
        ImFont *font = io.Fonts->AddFontDefault(&config);
        return font;
    }

};