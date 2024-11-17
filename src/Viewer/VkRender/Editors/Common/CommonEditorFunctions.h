//
// Created by mgjer on 12/10/2024.
//

#ifndef COMMONEDITORFUNCTIONS_H
#define COMMONEDITORFUNCTIONS_H

#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/ImGui/LayerUtils.h"

namespace VkRender::EditorUtils {

    // TODO dont know where to put this utility function
    static uint32_t getBytesPerPixelFromVkFormat(VkFormat format) {
        switch (format) {
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            return 4; // 4 bytes per pixel (8 bits per channel, 4 channels)
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SRGB:
            return 3; // 3 bytes per pixel (8 bits per channel, 3 channels)
        case VK_FORMAT_R8_UNORM:
            return 1; // 1 byte per pixel (8 bits per channel)
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return 8; // 8 bytes per pixel (16 bits per channel, 4 channels)
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            return 16; // 16 bytes per pixel (32 bits per channel, 4 channels)
        case VK_FORMAT_D32_SFLOAT:
            return 4; // 4 bytes per pixel for depth
        case VK_FORMAT_D24_UNORM_S8_UINT:
            return 4; // Approximation: 24 bits depth + 8 bits stencil = 4 bytes
        default:
            throw std::runtime_error("Unsupported VkFormat for size calculation");
        }
    }

    static std::shared_ptr<VulkanTexture2D> createEmptyTexture(uint32_t width, uint32_t height, VkFormat format,
                                                               Application* context, bool setValues = false) {
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
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
        imageViewCI.format = format;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(context->vkDevice(), context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vulkanImageCreateInfo.debugInfo = "Emtpy texture:" + std::to_string(width) + "x" + std::to_string(height);
        VulkanTexture2DCreateInfo textureCreateInfo(context->vkDevice());
        textureCreateInfo.image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
        auto texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        if (setValues) {
            uint32_t imageSize = static_cast<uint32_t>(width) * static_cast<uint32_t>(height) * getBytesPerPixelFromVkFormat(format);

            void* imageMemory = malloc(imageSize);
            std::fill(static_cast<uint8_t*>(imageMemory), static_cast<uint8_t*>(imageMemory) + imageSize, 200);
            texture->loadImage(imageMemory, imageSize);
            free(imageMemory);
        }

        return texture;
    }

    static std::shared_ptr<VulkanTexture2D>
    createTextureFromFile(std::filesystem::path filePath, Application* context) {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(filePath.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4; // Assuming STBI_rgb_alpha gives us 4 channels per pixel
        if (!pixels) {
            if (!pixels) {
                // load empty texture
                Log::Logger::getInstance()->error("Failed to load texture image: {}. Reverting to empty",
                                                  filePath.string());
                pixels = stbi_load((Utils::getTexturePath() / "moon.png").string().c_str(), &texWidth, &texHeight,
                                   &texChannels, STBI_rgb_alpha);
                imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * texChannels);
                if (!pixels) {
                    throw std::runtime_error("Failed to load backup texture image");
                }
            }
        }

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

        VulkanImageCreateInfo vulkanImageCreateInfo(context->vkDevice(), context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.debugInfo = "Color texture: Image Editor";
        VulkanTexture2DCreateInfo textureCreateInfo(context->vkDevice());
        textureCreateInfo.image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
        auto texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        // Copy data to texturere
        texture->loadImage(pixels, imageSize);
        // Free the image data
        stbi_image_free(pixels);
        return texture;
    }

    static void openImportFileDialog(const std::string& fileDescription,
                                     const std::vector<std::string>& type,
                                     LayerUtils::FileTypeLoadFlow flow,
                                     std::future<LayerUtils::LoadFileInfo>* loadFileFuture,
                                     std::filesystem::path openLocation = Utils::getSystemHomePath()) {
        if (!loadFileFuture->valid()) {
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLocation = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }
            *loadFileFuture = std::async(VkRender::LayerUtils::selectFile, "Select " + fileDescription + " file",
                                         type, openLocation, flow);
        }
    }

    static void openImportFolderDialog(const std::string& fileDescription,
                                       const std::vector<std::string>& type,
                                       LayerUtils::FileTypeLoadFlow flow,
                                       std::future<LayerUtils::LoadFileInfo>* loadFolderFuture) {
        if (!loadFolderFuture->valid()) {
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            std::string openLoc = Utils::getSystemHomePath().string();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLoc = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }
            *loadFolderFuture = std::async(VkRender::LayerUtils::selectFolder, "Select Folder", openLoc, flow);
        }
    }

    static void saveFileDialog(const std::string& fileDescription,
                               const std::vector<std::string>& type,
                               LayerUtils::FileTypeLoadFlow flow,
                               std::future<LayerUtils::LoadFileInfo>* loadFolderFuture,
                               std::filesystem::path openLocation = Utils::getSystemHomePath()) {
        if (!loadFolderFuture->valid()) {
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLocation = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }

            *loadFolderFuture = std::async(VkRender::LayerUtils::saveFile, "Save as", openLocation, flow);
        }
    }
}

#endif //COMMONEDITORFUNCTIONS_H
