//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_UTILS_H
#define MULTISENSE_UTILS_H


#include <fstream>
#include <vector>
#include <map>
#include "cassert"
#include "iostream"

#include "vulkan/vulkan_core.h"
#include "MultiSense/src/core/Definitions.h"
#include "Logger.h"
#include "MultiSense/src/imgui/Layer.h"

namespace Utils {

    static std::string getShadersPath() {
        return "Assets/Shaders/";
    }

    static std::string getAssetsPath() {
        return "Assets/";
    }

    static std::string getTexturePath() {
        return "Assets/Textures/";
    }

    static std::string getScriptsPath() {
        return "scripts/";
    }

    inline bool findValIfExists(std::map<int, AR::StreamingModes> map, StreamIndex streamIndex) {
        if (map.find(streamIndex) == map.end()) {
            Log::Logger::getInstance()->info("Could not find {} in stream map", (uint32_t) streamIndex);
            return false;
        }
        return true;
    }

    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool isInVector(std::vector<std::string> v, const std::string &str) {
        if (std::find(v.begin(), v.end(), str) != v.end())
            return true;
        return false;

    }

    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool removeFromVector(std::vector<std::string>* v, const std::string &str) {
        auto it = std::find(v->begin(), v->end(), str);
        if (it == v->end())
            return false;
        v->erase(it);

        return true;
    }


    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool delFromVector(std::vector<std::string> v, const std::string &str) {
        auto itr = std::find(v.begin(), v.end(), str);
        if (itr != v.end()) {
            v.erase(itr);
            return true;
        }
        return false;
    }

#ifdef WIN32
    inline BOOL hasAdminRights() {
        BOOL fRet = FALSE;
        HANDLE hToken = NULL;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
            TOKEN_ELEVATION Elevation;
            DWORD cbSize = sizeof(TOKEN_ELEVATION);
            if (GetTokenInformation(hToken, TokenElevation, &Elevation, sizeof(Elevation), &cbSize)) {
                fRet = Elevation.TokenIsElevated;
            }
        }
        if (hToken) {
            CloseHandle(hToken);
        }
        return !fRet;
    }
#endif


    inline CRLCameraResolution stringToCameraResolution(const std::string &resolution) {
        if (resolution == "960 x 600 x 64x") return CRL_RESOLUTION_960_600_64;
        if (resolution == "960 x 600 x 128x") return CRL_RESOLUTION_960_600_128;
        if (resolution == "960 x 600 x 256x") return CRL_RESOLUTION_960_600_256;
        if (resolution == "1920 x 1200 x 64x") return CRL_RESOLUTION_1920_1200_64;
        if (resolution == "1920 x 1200 x 128x") return CRL_RESOLUTION_1920_1200_128;
        if (resolution == "1920 x 1200 x 256x") return CRL_RESOLUTION_1920_1200_256;
        return CRL_RESOLUTION_NONE;
    }

    /** @brief Convert camera resolution enum to uint32_t values used by the libmultisense */
    inline void
    cameraResolutionToValue(CRLCameraResolution resolution, uint32_t *_width, uint32_t *_height, uint32_t *_depth) {
        uint32_t width, height, depth;
        switch (resolution) {
            case CRL_RESOLUTION_NONE:
                width = 0;
                height = 0;
                depth = 0;
                break;
            case CRL_RESOLUTION_960_600_64:
                width = 960;
                height = 600;
                depth = 64;
                break;
            case CRL_RESOLUTION_960_600_128:
                width = 960;
                height = 600;
                depth = 128;
                break;
            case CRL_RESOLUTION_960_600_256:
                width = 960;
                height = 600;
                depth = 256;
                break;
            case CRL_RESOLUTION_1920_1200_64:
                width = 1920;
                height = 1200;
                depth = 64;
                break;
            case CRL_RESOLUTION_1920_1200_128:
                width = 1920;
                height = 1200;
                depth = 128;
                break;
            case CRL_RESOLUTION_1920_1200_256:
                width = 1920;
                height = 1200;
                depth = 256;
                break;
        }

        *_width = width;
        *_height = height;
        *_depth = depth;
    }

    inline VkFormat
    findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &candidates, VkImageTiling tiling,
                        VkFormatFeatureFlags features) {
        for (VkFormat format: candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    inline VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
        return findSupportedFormat(physicalDevice,
                                   {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }


    // Create an image memory barrier for changing the layout of
    // an image and put it into an active command buffer
    // See chapter 11.4 "Image Layout" for details

    inline void setImageLayout(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkImageSubresourceRange subresourceRange,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask) {
        // Create an image barrier object
        VkImageMemoryBarrier imageMemoryBarrier{};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        imageMemoryBarrier.oldLayout = oldImageLayout;
        imageMemoryBarrier.newLayout = newImageLayout;
        imageMemoryBarrier.image = image;
        imageMemoryBarrier.subresourceRange = subresourceRange;

        // Source layouts (old)
        // Source access mask controls actions that have to be finished on the old layout
        // before it will be transitioned to the new layout
        switch (oldImageLayout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                // Image layout is undefined (or does not matter)
                // Only valid as initial layout
                // No flags required, listed only for completeness
                imageMemoryBarrier.srcAccessMask = 0;
                break;

            case VK_IMAGE_LAYOUT_PREINITIALIZED:
                // Image is preinitialized
                // Only valid as initial layout for linear images, preserves memory contents
                // Make sure host writes have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                // Image is a color attachment
                // Make sure any writes to the color buffer have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                // Image is a depth/stencil attachment
                // Make sure any writes to the depth/stencil buffer have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                // Image is a transfer source
                // Make sure any reads from the image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                // Image is a transfer destination
                // Make sure any writes to the image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                // Image is read by a shader
                // Make sure any shader reads from the image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;
            default:
                // Other source layouts aren't firstUpdate (yet)
                break;
        }

        // Target layouts (new)
        // Destination access mask controls the dependency for the new image layout
        switch (newImageLayout) {
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                // Image will be used as a transfer destination
                // Make sure any writes to the image have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                // Image will be used as a transfer source
                // Make sure any reads from the image have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                // Image will be used as a color attachment
                // Make sure any writes to the color buffer have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                // Image layout will be used as a depth/stencil attachment
                // Make sure any writes to depth/stencil buffer have been finished
                imageMemoryBarrier.dstAccessMask =
                        imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                // Image will be read in a shader (sampler, input attachment)
                // Make sure any writes to the image have been finished
                if (imageMemoryBarrier.srcAccessMask == 0) {
                    imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                }
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;
            default:
                // Other source layouts aren't firstUpdate (yet)
                break;
        }

        // Put barrier inside setup command buffer
        vkCmdPipelineBarrier(
                cmdbuffer,
                srcStageMask,
                dstStageMask,
                0,
                0, nullptr,
                0, nullptr,
                1, &imageMemoryBarrier);
    }

    // Fixed sub resource on first mip level and layer
    inline void setImageLayout(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkImageAspectFlags aspectMask,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask) {
        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = aspectMask;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 1;
        setImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange, srcStageMask, dstStageMask);
    }

    inline void
    copyBufferToImage(VkCommandBuffer cmdBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height,
                      VkImageAspectFlagBits aspectFlagBits) {
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask = aspectFlagBits;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageSubresource.mipLevel = 0;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageExtent.width = width;
        bufferCopyRegion.imageExtent.height = height;
        bufferCopyRegion.imageExtent.depth = 1;

        vkCmdCopyBufferToImage(
                cmdBuffer,
                buffer,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &bufferCopyRegion
        );

    }

    inline VkShaderModule loadShader(const char *fileName, VkDevice device) {
        std::ifstream is(fileName, std::ios::binary | std::ios::ate);

        if (is.is_open()) {
            size_t size = is.tellg();
            is.seekg(0, std::ios::beg);
            char *shaderCode = new char[size];
            is.read(shaderCode, size);
            is.close();

            assert(size > 0);

            VkShaderModule shaderModule;
            VkShaderModuleCreateInfo moduleCreateInfo{};
            moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleCreateInfo.codeSize = size;
            moduleCreateInfo.pCode = (uint32_t *) shaderCode;

            if (vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule) != VK_SUCCESS)
                throw std::runtime_error("Failed to crate shader module");


            delete[] shaderCode;

            return shaderModule;
        } else {
            std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << "\n";
            return VK_NULL_HANDLE;
        }
    }
}

#endif //MULTISENSE_UTILS_H
