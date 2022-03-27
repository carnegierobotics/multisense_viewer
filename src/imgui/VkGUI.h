//
// Created by magnus on 2/4/22.
//

#ifndef MULTISENSE_VKGUI_H
#define MULTISENSE_VKGUI_H


#include <vulkan/vulkan.h>
#include <MultiSense/src/core/VulkanDevice.h>
#include <MultiSense/src/core/VulkanRenderer.h>
#include <MultiSense/src/core/Buffer.h>
#include <MultiSense/external/imgui/imgui.h>
#include <MultiSense/src/tools/Utils.h>
#include "UISettings.h"
#include "imgui_internal.h"


class ImGUI {
private:
    // Vulkan resources for rendering the UI
    VkSampler sampler{};
    Buffer vertexBuffer;
    Buffer indexBuffer;
    int32_t vertexCount = 0;
    int32_t indexCount = 0;
    VkDeviceMemory fontMemory = VK_NULL_HANDLE;
    VkImage fontImage = VK_NULL_HANDLE;
    VkImageView fontView = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
    VkDescriptorPool descriptorPool{};
    VkDescriptorSetLayout descriptorSetLayout{};
    VkDescriptorSet descriptorSet{};

    VulkanDevice *device;

public:
    // UI params are set via push constants
    struct PushConstBlock {
        glm::vec2 scale;
        glm::vec2 translate;
    } pushConstBlock{};

    std::vector<VkPipelineShaderStageCreateInfo> shaders;

    UISettings *uiSettings;
    bool updated = false;
    /** @brief active used to determine if camera should update view or user is currently using the UI */
    bool active = false;
    bool firstUpdate = true;

    explicit ImGUI(VulkanDevice *vulkanDevice) {
        device = vulkanDevice;
        ImGui::CreateContext();

        uiSettings = new UISettings();
        ImGuiStyle *style = &ImGui::GetStyle();

        /*
         style->WindowPadding = ImVec2(15, 15);
        style->WindowRounding = 5.0f;
        style->FramePadding = ImVec2(5, 5);
        style->FrameRounding = 4.0f;
        style->ItemSpacing = ImVec2(12, 8);
        style->ItemInnerSpacing = ImVec2(8, 6);
        style->IndentSpacing = 25.0f;
        style->ScrollbarSize = 15.0f;
        style->ScrollbarRounding = 9.0f;
        style->GrabMinSize = 5.0f;
        style->GrabRounding = 3.0f;

        style->Colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.83f, 1.00f);
        style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
        style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
        style->Colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
        style->Colors[ImGuiCol_Border] = ImVec4(0.80f, 0.80f, 0.83f, 0.88f);
        style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.92f, 0.91f, 0.88f, 0.00f);
        style->Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
        style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
        style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
        style->Colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
        style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 0.98f, 0.95f, 0.75f);
        style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
        style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
        style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
        style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
        style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
        style->Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
        style->Colors[ImGuiCol_CheckMark] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
        style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
        style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
        style->Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
        style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
        style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
        style->Colors[ImGuiCol_Header] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
        style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
        style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);

        style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
        style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
        style->Colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
        style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
        style->Colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
        style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
        style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.25f, 1.00f, 0.00f, 0.43f);
         */

    };

    ~ImGUI() {
        ImGui::DestroyContext();
        // Release all Vulkan resources required for rendering imGui
        vertexBuffer.destroy();
        indexBuffer.destroy();
        vkDestroyImage(device->logicalDevice, fontImage, nullptr);
        vkDestroyImageView(device->logicalDevice, fontView, nullptr);
        vkFreeMemory(device->logicalDevice, fontMemory, nullptr);
        vkDestroySampler(device->logicalDevice, sampler, nullptr);
        vkDestroyPipelineCache(device->logicalDevice, pipelineCache, nullptr);
        vkDestroyPipeline(device->logicalDevice, pipeline, nullptr);
        vkDestroyPipelineLayout(device->logicalDevice, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(device->logicalDevice, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device->logicalDevice, descriptorSetLayout, nullptr);
    }

    // Initialize styles, keys, etc.
    void init(float width, float height) {
        // Color scheme
        ImGuiStyle &style = ImGui::GetStyle();
        style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
        // Dimensions
        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize = ImVec2(width, height);
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    }


    // Initialize all Vulkan resources used by the ui
    // Graphics pipeline
    void initResources(VkRenderPass renderPass, VkQueue copyQueue, const std::string &shadersPath) {

        ImGuiIO &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        // Create font texture
        unsigned char *fontData;
        int texWidth, texHeight;
        io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);
        VkDeviceSize uploadSize = texWidth * texHeight * 4 * sizeof(char);

        // Create Target Image for copy
        VkImageCreateInfo imageInfo = Populate::imageCreateInfo();
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent.width = texWidth;
        imageInfo.extent.height = texHeight;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device->logicalDevice, &imageInfo, nullptr, &fontImage) != VK_SUCCESS)
            throw std::runtime_error("Failed to create ImGUI image");
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device->logicalDevice, fontImage, &memReqs);
        VkMemoryAllocateInfo memAllocInfo = Populate::memoryAllocateInfo();
        memAllocInfo.allocationSize = memReqs.size;
        memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device->logicalDevice, &memAllocInfo, nullptr, &fontMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate image IMGUI memory");
        if (vkBindImageMemory(device->logicalDevice, fontImage, fontMemory, 0))
            throw std::runtime_error("Failed to bind imgui image with its memory");

        // Create Image view to interface with image.
        // Image view
        VkImageViewCreateInfo viewInfo = Populate::imageViewCreateInfo();
        viewInfo.image = fontImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(device->logicalDevice, &viewInfo, nullptr, &fontView) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image view for imgui image");
        // Staging buffers for font upload
        Buffer stagingBuffer;
        // Copy data to staging buffers
        if (device->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 &stagingBuffer, uploadSize) != VK_SUCCESS)
            throw std::runtime_error("Failed to create staging buffer");
        stagingBuffer.map();
        memcpy(stagingBuffer.mapped, fontData, uploadSize);
        stagingBuffer.unmap();

        // Transition image layout to transfer and specify copy region
        // Copy buffer data to font image
        VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        // Prepare for transfer
        Utils::setImageLayout(
                copyCmd,
                fontImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Copy
        // Copy staging buffer to image
        Utils::copyBufferToImage(copyCmd, stagingBuffer.buffer, fontImage, texWidth, texHeight,
                                 VK_IMAGE_ASPECT_COLOR_BIT);

        // Prepare image for shader read
        // Prepare for shader read
        Utils::setImageLayout(
                copyCmd,
                fontImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        device->flushCommandBuffer(copyCmd, copyQueue, true);

        stagingBuffer.destroy();
        // submit cmd buffer and destroy staging buffer

        // Texture sampler
        // Font texture Sampler
        VkSamplerCreateInfo samplerInfo = Populate::samplerCreateInfo();
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        if (vkCreateSampler(device->logicalDevice, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture sampler IMGUI");
        // Descriptor pool
        // Descriptor layout
        // Descriptor set

        // Descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes = {
                Populate::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = Populate::descriptorPoolCreateInfo(poolSizes, 2);
        if (vkCreateDescriptorPool(device->logicalDevice, &descriptorPoolInfo, nullptr, &descriptorPool) !=
            VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool");

        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                Populate::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                     VK_SHADER_STAGE_FRAGMENT_BIT, 0),
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout = Populate::descriptorSetLayoutCreateInfo(setLayoutBindings);
        if (vkCreateDescriptorSetLayout(device->logicalDevice, &descriptorLayout, nullptr, &descriptorSetLayout) !=
            VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout");

        // Descriptor set
        VkDescriptorSetAllocateInfo allocInfo = Populate::descriptorSetAllocateInfo(descriptorPool,
                                                                                    &descriptorSetLayout, 1);
        if (vkAllocateDescriptorSets(device->logicalDevice, &allocInfo, &descriptorSet) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set");
        VkDescriptorImageInfo fontDescriptor = Populate::descriptorImageInfo(
                sampler,
                fontView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                Populate::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0,
                                             &fontDescriptor)
        };
        vkUpdateDescriptorSets(device->logicalDevice, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, nullptr);


        // Pipeline ---------

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

        if (vkCreateGraphicsPipelines(device->logicalDevice, pipelineCache, 1, &pipelineCreateInfo, nullptr,
                                      &pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");
    }


    void createDropDowns(ElementBase *element) {
        if (ImGui::BeginCombo(element->dropDown->label.c_str(),
                              element->dropDown->selected.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto &dropDownItem: element->dropDown->dropDownItems) {

                bool is_selected = (element->dropDown->selected ==
                                    dropDownItem); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(dropDownItem.c_str(), is_selected)) {
                    element->dropDown->selected = dropDownItem;
                    updated |= true;
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
            }
            ImGui::EndCombo();
        }
    }

    void buildElements(ElementBase element) {
        ImGui::SetCursorPos(ImVec2(element.pos.x, element.pos.y));

        switch (element.type) {
            case AR_ELEMENT_TEXT:
                ImGui::TextColored(element.text->color, "%s", element.text->string.c_str());

                if (element.text->sameLine)
                    ImGui::SameLine();

                break;
            case AR_ELEMENT_BUTTON:
                element.button->clicked = ImGui::Button(element.button->string.c_str(), element.button->size);
                if (element.button->clicked)
                    updated |= true;
                break;
            case AR_ELEMENT_FLOAT_SLIDER:
                break; // TODO IMPLEMENT
            case AR_UI_ELEMENT_DROPDOWN:
                createDropDowns(&element);
                break;
            default:
                std::cerr << "Gui element not supported in AR Renderer\n";
                break;
        }
    }

// Starts a new imGui frame and sets up windows and ui elements
    void newFrame(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height) {

        ImGui::NewFrame();
        updated = false;

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        float sidebarWidth = 250.0f;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(sidebarWidth, (float) height));
        ImGui::Begin("SideBar", &pOpen, window_flags);

        auto *wnd = ImGui::FindWindowByName("GUI");
        if (wnd) {
            ImGuiDockNode *node = wnd->DockNode;
            if (node)
                node->WantHiddenTabBarToggle = true;

        }

        ImGui::TextUnformatted("title");
        ImGui::TextUnformatted(device->properties.deviceName);

        // Update frame time display
        if (updateFrameGraph) {
            std::rotate(uiSettings->frameTimes.begin(), uiSettings->frameTimes.begin() + 1,
                        uiSettings->frameTimes.end());
            float frameTime = 1000.0f / (frameTimer * 1000.0f);
            uiSettings->frameTimes.back() = frameTime;
            if (frameTime < uiSettings->frameTimeMin) {
                uiSettings->frameTimeMin = frameTime;
            }
            if (frameTime > uiSettings->frameTimeMax) {
                uiSettings->frameTimeMax = frameTime;
            }
        }

        ImGui::PlotLines("Frame Times", &uiSettings->frameTimes[0], 50, 0, "", uiSettings->frameTimeMin,
                         uiSettings->frameTimeMax, ImVec2(0, 80));


        static int item_current_idx = 0; // Here we store our selection data as an index.
        ImGui::Spacing();
        ImGui::Text("Active objects");
        if (ImGui::BeginListBox("##")) {
            for (int n = 0; n < uiSettings->listBoxNames.size(); n++) {
                const bool is_selected = (uiSettings->selectedListboxIndex == n);

                // Remove Example from list
                if (uiSettings->listBoxNames[n] == "Example") continue;

                if (ImGui::Selectable(uiSettings->listBoxNames[n].c_str(), is_selected)) {
                    uiSettings->selectedListboxIndex = n;

                }
                // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }

            }
            ImGui::EndListBox();
        }

        if (ImGui::BeginPopupModal("add_device_modal", NULL,
                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
            if (!uiSettings->modalElements.empty()) {

                auto &btn = uiSettings->modalElements[0];
                auto &inputName = uiSettings->modalElements[1];
                auto &inputIP = uiSettings->modalElements[2];

                ImGui::Text("Connect to your MultiSense Device");
                ImGui::Separator();
                ImGui::InputText("Profile name", inputName.inputText->string,
                                 IM_ARRAYSIZE(inputName.inputText->string));
                ImGui::InputText("Camera ip", inputIP.inputText->string, IM_ARRAYSIZE(inputIP.inputText->string));
                btn.button->clicked = ImGui::Button( btn.button->string.c_str(),  btn.button->size);

                if (uiSettings->closeModalPopup)
                    ImGui::CloseCurrentPopup();

                updated |= true;

                ImGui::EndPopup();
            }
        }




        if (!uiSettings->elements.empty()) {
            for (auto &element: uiSettings->elements) {
                if (element.location == "sidebar")
                    buildElements(element);

                // Treat special events from ImGUI
                if (element.button && element.button->string == "ADD DEVICE" && element.button->clicked) {
                    ImGui::OpenPopup("add_device_modal");
                }

            }
        }
        if (!uiSettings->modalElements.empty()) {
            auto &element = uiSettings->modalElements;

        }


        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2((float) width - sidebarWidth, height - 550));
        ImGui::Begin("main", &pOpen, window_flags);

        if (!uiSettings->elements.empty()) {
            for (auto &element: uiSettings->elements) {
                if (element.location == "main")
                    buildElements(element);
            }
        }
        ImGui::End();

        active = ImGui::IsWindowHovered() || ImGui::IsAnyItemHovered() || ImGui::IsAnyItemFocused();

        //ImGui::ShowDemoWindow();
        ImGui::Render();
    }


// Update vertex and index buffer containing the imGui elements when required
    bool updateBuffers() {
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

// Draw current imGui frame into a command buffer
    void drawFrame(VkCommandBuffer commandBuffer) {
        ImGuiIO &io = ImGui::GetIO();

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0,
                                nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        //VkViewport viewport = Populate
        // ::viewport(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f);
        //vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        // UI scale and translate via push constants
        pushConstBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
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


    bool checkBox(const char *caption, bool *value) {
        bool res = ImGui::Checkbox(caption, value);
        if (res) { updated = true; };
        return res;
    }

    bool checkBox(const char *caption, int32_t *value) {
        bool val = (*value == 1);
        bool res = ImGui::Checkbox(caption, &val);
        *value = val;
        if (res) { updated = true; };
        return res;
    }

};


#endif //MULTISENSE_VKGUI_H
