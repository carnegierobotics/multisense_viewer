//
// Created by magnus on 7/30/24.
//

#include <stb_image.h>
#include "Viewer/VkRender/ImGui/GuiAssets.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"
#include "Viewer/Application/Application.h"


namespace VkRender {

    GuiAssets::GuiAssets(Application *context) : device(context->vkDevice()) {

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
                vkCreateDescriptorSetLayout(device.m_LogicalDevice, &layoutCreateInfo, nullptr,
                                            &descriptorSetLayout));


        VkDescriptorPoolCreateInfo poolCreateInfo = Populate::descriptorPoolCreateInfo(poolSizes, setCount);
        CHECK_RESULT(vkCreateDescriptorPool(device.m_LogicalDevice, &poolCreateInfo, nullptr, &descriptorPool));


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
        font13 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 13.0f, false, context);
        font8 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 8.0f, false, context);
        font15 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 15.0f, false, context);
        font18 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 18.0f, false, context);
        font24 = loadFontFromFileName("Assets/Fonts/Roboto-Black.ttf", 24.0f, false, context);

        fontIcons = loadFontFromFileName("Assets/Fonts/fa-solid-900.ttf", 13.0f, true, context);
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
        //iconTextures.reserve(iconFileNames.size());
        iconCount = iconFileNames.size() - 1;
        imageIconDescriptors.resize(iconFileNames.size());
        // Base path for the texture files
        std::filesystem::path texturePath = Utils::getTexturePath();
        // Load textures using the filenames
        for (std::size_t index = 0; index < iconFileNames.size(); ++index) {
            const auto &filename = iconFileNames[index];
            loadImGuiTextureFromFileName((texturePath / filename).string(), index, context);
        }

        //loadAnimatedGif(Utils::getTexturePath().append("spinner.gif").string());

        // setup graphics pipeline
        VkShaderModule vtxModule{};
        Utils::loadShader((Utils::getShadersPath().append("Scene/imgui/ui.vert.spv")).string().c_str(),
                          device.m_LogicalDevice, &vtxModule);
        VkPipelineShaderStageCreateInfo vtxShaderStage = {};
        vtxShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vtxShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vtxShaderStage.module = vtxModule;
        vtxShaderStage.pName = "main";
        assert(vtxShaderStage.module != VK_NULL_HANDLE);
        shaderModules.push_back(vtxModule);

        VkShaderModule frgModule;
        Utils::loadShader((Utils::getShadersPath().append("Scene/imgui/ui.frag.spv")).string().c_str(),
                          device.m_LogicalDevice, &frgModule);
        VkPipelineShaderStageCreateInfo fragShaderStage = {};
        fragShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStage.module = frgModule;
        fragShaderStage.pName = "main";
        assert(fragShaderStage.module != VK_NULL_HANDLE);
        shaderModules.push_back(frgModule);

        shaders = std::vector<VkPipelineShaderStageCreateInfo>{vtxShaderStage, fragShaderStage};



    }


    ImFont *GuiAssets::loadFontFromFileName(const std::filesystem::path &file, float fontSize, bool iconFont, Application* context) {
        ImFont *font;

        if (iconFont) {
            float baseFontSize = fontSize; // 13.0f is the size of the default font. Change to the font size you use.
            float iconFontSize = baseFontSize * 2.0f /3.0f; // FontAwesome fonts need to have their sizes reduced by 2.0f/3.0f in order to align correctly

            // merge in icons from Font Awesome
            static const ImWchar icons_ranges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
            ImFontConfig icons_config;
            //icons_config.MergeMode = true;
            //icons_config.PixelSnapH = true;
            icons_config.GlyphMinAdvanceX = 13.0f;
            font = fontAtlas.AddFontFromFileTTF(file.string().c_str(), baseFontSize, &icons_config, icons_ranges);
        } else {
            ImFontConfig config;
            config.OversampleH = 2;
            config.OversampleV = 1;
            config.GlyphExtraSpacing.x = 1.0f;
            font = fontAtlas.AddFontFromFileTTF(file.string().c_str(), fontSize, &config);
        }


        unsigned char *pixels;
        int width, height;
        fontAtlas.GetTexDataAsRGBA32(&pixels, &width, &height);
        auto uploadSize = width * height * 4 * sizeof(char);

        // Setup default texture
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageCI.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(device, context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vulkanImageCreateInfo.debugInfo = "GuiResources:FontTexture";
        auto image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);

        VulkanTexture2DCreateInfo textureCreateInfo(device);
        textureCreateInfo.image = image;
        std::shared_ptr<VulkanTexture2D> texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        texture->loadImage(pixels, width*height * 4);
        fontTexture.emplace_back(std::move(texture));

        VkDescriptorSet descriptor{};
        // descriptors
        // Create Descriptor Set:
        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            VkResult res = vkAllocateDescriptorSets(device.m_LogicalDevice, &alloc_info, &descriptor);
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
            write_desc[0].pImageInfo = &fontTexture.back()->getDescriptorInfo();
            vkUpdateDescriptorSets(device.m_LogicalDevice, 1, write_desc, 0, nullptr);
        }


        fontDescriptors.push_back(descriptor);
        return font;
    }


    void GuiAssets::loadAnimatedGif(const std::string &file) {
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
/*
        for (int i = 0; i < depth; ++i) {
            VkDescriptorSet dSet{};

            gifTexture[i] = std::make_unique<Texture2D>(pixelPointer, imageSize, VK_FORMAT_R8G8B8A8_SRGB,
                                                        width, height, device,
                                                        device.m_TransferQueue, VK_FILTER_LINEAR,
                                                        VK_IMAGE_USAGE_SAMPLED_BIT,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            // Create Descriptor Set:

            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            CHECK_RESULT(vkAllocateDescriptorSets(device.m_LogicalDevice, &alloc_info, &dSet));


            // Update the Descriptor Set:
            VkWriteDescriptorSet write_desc[1] = {};
            write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_desc[0].dstSet = dSet;
            write_desc[0].descriptorCount = 1;
            write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_desc[0].pImageInfo = &gifTexture[i]->m_descriptor;
            vkUpdateDescriptorSets(device.m_LogicalDevice, 1, write_desc, 0, NULL);
            pixelPointer += imageSize;

            gifImageDescriptors.emplace_back(dSet);
        }
        stbi_image_free(pixels);
        */
    }

    void GuiAssets::loadImGuiTextureFromFileName(const std::string &file, uint32_t i, Application* context) {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(file.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * texChannels);
        if (!pixels) {
            throw std::runtime_error("failed to load texture m_Image: " + file);
        }

        // Setup default texture
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageCI.extent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(device, context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vulkanImageCreateInfo.debugInfo = "GuiResources:IconTexture";
        auto image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);

        VulkanTexture2DCreateInfo textureCreateInfo(device);
        textureCreateInfo.image = image;
        std::shared_ptr<VulkanTexture2D> texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        texture->loadImage(pixels, texWidth*texHeight*4);
        iconTextures.emplace_back(std::move(texture));

        /*
        iconTextures.emplace_back(pixels, imageSize, VK_FORMAT_R8G8B8A8_SRGB, static_cast<uint32_t>(texWidth),
                                  static_cast<uint32_t>(texHeight), device,
                                  device.m_TransferQueue, VK_FILTER_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        */


        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptorPool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptorSetLayout;
            CHECK_RESULT(vkAllocateDescriptorSets(device.m_LogicalDevice, &alloc_info, &imageIconDescriptors[i]));
        }
        // Update the Descriptor Set:
        {

            VkWriteDescriptorSet write_desc[1] = {};
            write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_desc[0].dstSet = imageIconDescriptors[i];
            write_desc[0].descriptorCount = 1;
            write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_desc[0].pImageInfo = &iconTextures[i]->getDescriptorInfo();
            vkUpdateDescriptorSets(device.m_LogicalDevice, 1, write_desc, 0, NULL);
        }


    }

}