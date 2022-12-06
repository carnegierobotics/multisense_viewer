//
// Created by magnus on 9/4/21.
//

#ifndef MULTISENSE_VIEWER_UTILS_H
#define MULTISENSE_VIEWER_UTILS_H

#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>
#include <vulkan/vulkan_core.h>

#ifdef WIN32
#include <direct.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#else
#include <sys/stat.h>
#endif

#include "Viewer/Tools/Macros.h"
#include "Viewer/Core/Definitions.h"
#include "Viewer/Tools/Logger.h"

namespace Utils {
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FUNCTION

#ifdef __linux__
    static std::string getShadersPath() {
        return "./Assets/Shaders/";
    }
    static std::string getAssetsPath() {
        return "./Assets/";
    }

    static std::string getTexturePath() {
        return "./Assets/Textures/";
    }

    static std::string getScriptsPath() {
        return "Scripts/";
    }

#endif

#ifdef WIN32
    static std::string getShadersPath() {
        return ".\\Assets\\Shaders\\";
    }
    static std::string getAssetsPath() {
        return ".\\Assets\\";
    }

    static std::string getTexturePath() {
        return ".\\Assets\\Textures\\";
    }

    static std::string getScriptsPath() {
        return "Scripts\\";
    }
#endif

    static std::string dataSourceToString(crl::multisense::DataSource d) {
        switch (d) {
            case crl::multisense::Source_Raw_Left:
                return "Raw Left";
            case crl::multisense::Source_Raw_Right:
                return "Raw Right";
            case crl::multisense::Source_Luma_Left:
                return "Luma Left";
            case crl::multisense::Source_Luma_Right:
                return "Luma Right";
            case crl::multisense::Source_Luma_Rectified_Left:
                return "Luma Rectified Left";
            case crl::multisense::Source_Luma_Rectified_Right:
                return "Luma Rectified Right";
            case crl::multisense::Source_Chroma_Left:
                return "Color Left";
            case crl::multisense::Source_Chroma_Right:
                return "Source Color Right";
            case crl::multisense::Source_Disparity_Left:
                return "Disparity Left";
            case crl::multisense::Source_Disparity_Cost:
                return "Disparity Cost";
            case crl::multisense::Source_Lidar_Scan:
                return "Source Lidar Scan";
            case crl::multisense::Source_Raw_Aux:
                return "Raw Aux";
            case crl::multisense::Source_Luma_Aux:
                return "Luma Aux";
            case crl::multisense::Source_Luma_Rectified_Aux:
                return "Luma Rectified Aux";
            case crl::multisense::Source_Chroma_Aux:
                return "Color Aux";
            case crl::multisense::Source_Chroma_Rectified_Aux:
                return "Color Rectified Aux";
            case crl::multisense::Source_Disparity_Aux:
                return "Disparity Aux";
            case crl::multisense::Source_Compressed_Left:
                return "Luma Compressed Left";
            case crl::multisense::Source_Compressed_Rectified_Left:
                return "Luma Compressed Rectified Left";
            case crl::multisense::Source_Compressed_Right:
                return "Luma Compressed Right";
            case crl::multisense::Source_Compressed_Rectified_Right:
                return "Luma Compressed Rectified Reight";
            case crl::multisense::Source_Compressed_Aux:
                return "Compressed Aux";
            case crl::multisense::Source_Compressed_Rectified_Aux:
                return "Compressed Rectified Aux";
            case (crl::multisense::Source_Chroma_Rectified_Aux | crl::multisense::Source_Luma_Rectified_Aux):
                return "Color Rectified Aux";
            case (crl::multisense::Source_Chroma_Aux | crl::multisense::Source_Luma_Aux):
                return "Color Aux";
            case crl::multisense::Source_Imu:
                return "IMU";
            default:
                return "Unknown";
        }
    }

    static crl::multisense::DataSource stringToDataSource(const std::string &d) {
        if (d == "Raw Left") return crl::multisense::Source_Raw_Left;
        if (d == "Raw Right") return crl::multisense::Source_Raw_Right;
        if (d == "Luma Left") return crl::multisense::Source_Luma_Left;
        if (d == "Luma Right") return crl::multisense::Source_Luma_Right;
        if (d == "Luma Rectified Left") return crl::multisense::Source_Luma_Rectified_Left;
        if (d == "Luma Rectified Right") return crl::multisense::Source_Luma_Rectified_Right;
        if (d == "Luma Compressed Rectified Left") return crl::multisense::Source_Compressed_Rectified_Left;
        if (d == "Luma Compressed Left") return crl::multisense::Source_Compressed_Left;
        if (d == "Color Left") return crl::multisense::Source_Chroma_Left;
        if (d == "Source Color Right") return crl::multisense::Source_Chroma_Right;
        if (d == "Disparity Left") return crl::multisense::Source_Disparity_Left;
        if (d == "Disparity Cost") return crl::multisense::Source_Disparity_Cost;
        if (d == "Jpeg Left") return crl::multisense::Source_Jpeg_Left;
        if (d == "Source Rgb Left") return crl::multisense::Source_Rgb_Left;
        if (d == "Source Lidar Scan") return crl::multisense::Source_Lidar_Scan;
        if (d == "Raw Aux") return crl::multisense::Source_Raw_Aux;
        if (d == "Luma Aux") return crl::multisense::Source_Luma_Aux;
        if (d == "Luma Rectified Aux") return crl::multisense::Source_Luma_Rectified_Aux;
        if (d == "Color Aux") return crl::multisense::Source_Chroma_Aux;
        if (d == "Color Rectified Aux") return crl::multisense::Source_Chroma_Rectified_Aux;
        if (d == "Chroma Aux") return crl::multisense::Source_Chroma_Aux;
        if (d == "Chroma Rectified Aux") return crl::multisense::Source_Chroma_Rectified_Aux;
        if (d == "Disparity Aux") return crl::multisense::Source_Disparity_Aux;
        if (d == "IMU") return crl::multisense::Source_Imu;
        if (d == "All") return crl::multisense::Source_All;
        return false;
    }

    static CRLCameraDataType CRLSourceToTextureType(const std::string &d) {
        if (d == "Luma Left") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Right") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Rectified Left") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Rectified Right") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Left") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Right") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Rectified Left") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Rectified Right") return CRL_GRAYSCALE_IMAGE;
        if (d == "Disparity Left") return CRL_DISPARITY_IMAGE;

        if (d == "Color Aux") return CRL_COLOR_IMAGE_YUV420;
        if (d == "Color Rectified Aux") return CRL_COLOR_IMAGE_YUV420;
        if (d == "Luma Aux") return CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Rectified Aux") return CRL_GRAYSCALE_IMAGE;
        if (d == "Compressed Aux") return CRL_COLOR_IMAGE_YUV420;
        if (d == "Compressed Rectified Aux") return CRL_COLOR_IMAGE_YUV420;

        return CRL_CAMERA_IMAGE_NONE;
    }

    DISABLE_WARNING_POP


    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool isInVector(const std::vector<std::string> &v, const std::string &str) {
        if (std::find(v.begin(), v.end(), str) != v.end())
            return true;
        return false;

    }

    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool removeFromVector(std::vector<std::string> *v, const std::string &str) {
        auto it = std::find(v->begin(), v->end(), str);
        if (it == v->end())
            return false;
        v->erase(it);

        return true;
    }


    /**@brief small utility function. Usage of this makes other code more readable */
    template<typename T>
    inline bool delFromVector(std::vector<T> *v, const T &str) {
        auto itr = std::find(v->begin(), v->end(), str);
        if (itr != v->end()) {
            v->erase(itr);
            return true;
        }
        return false;
    }

#ifdef WIN32

    inline bool hasAdminRights() {
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
        if (resolution == "1024 x 1024 x 128x") return CRL_RESOLUTION_1024_1024_128;
        if (resolution == "2048 x 1088 x 256x") return CRL_RESOLUTION_2048_1088_256;
        return CRL_RESOLUTION_NONE;
    }

    inline std::string cameraResolutionToString(const CRLCameraResolution &res) {
        switch (res) {
            case CRL_RESOLUTION_960_600_64:
                return "960 x 600 x 64";
            case CRL_RESOLUTION_960_600_128:
                return "960 x 600 x 128";
            case CRL_RESOLUTION_960_600_256:
                return "960 x 600 x 256";
            case CRL_RESOLUTION_1920_1200_64:
                return "1920 x 1200 x 64";
            case CRL_RESOLUTION_1920_1200_128:
                return "1920 x 1200 x 128";
            case CRL_RESOLUTION_1920_1200_256:
                return "1920 x 1200 x 256";
            case CRL_RESOLUTION_1024_1024_128:
                return "1024 x 1024 x 128";
            case CRL_RESOLUTION_2048_1088_256:
                return "2048 x 1088 x 256";
            case CRL_RESOLUTION_NONE:
                return "Resolution not supported";
        }
        return "Resolution not supported";
    }

    /** @brief Convert camera resolution enum to uint32_t values used by the libmultisense */
    inline void
    cameraResolutionToValue(CRLCameraResolution resolution, uint32_t *_width, uint32_t *_height, uint32_t *_depth) {
        uint32_t width = 0, height = 0, depth = 0;
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
            case CRL_RESOLUTION_1024_1024_128:
                width = 1024;
                height = 1024;
                depth = 128;
                break;
            case CRL_RESOLUTION_2048_1088_256:
                width = 2048;
                height = 1088;
                depth = 256;
                break;
        }

        *_width = width;
        *_height = height;
        *_depth = depth;
    }

    /** @brief Convert camera resolution enum to uint32_t values used by the libmultisense */
    inline CRLCameraResolution
    valueToCameraResolution(uint32_t _width, uint32_t _height, uint32_t _depth) {
        if (_height == 600 && _width == 960 && _depth == 64) {
            return CRL_RESOLUTION_960_600_64;
        }
        if (_height == 600 && _width == 960 && _depth == 128) {
            return CRL_RESOLUTION_960_600_128;
        }
        if (_height == 600 && _width == 960 && _depth == 256) {
            return CRL_RESOLUTION_960_600_256;
        }
        if (_height == 1200 && _width == 1920 && _depth == 64) {
            return CRL_RESOLUTION_1920_1200_64;
        }
        if (_height == 1200 && _width == 1920 && _depth == 128) {
            return CRL_RESOLUTION_1920_1200_128;
        }
        if (_height == 1200 && _width == 1920 && _depth == 256) {
            return CRL_RESOLUTION_1920_1200_256;
        }
        if (_height == 1024 && _width == 1024 && _depth == 128) {
            return CRL_RESOLUTION_1024_1024_128;
        }
        if (_height == 2048 && _width == 1088 && _depth == 256) {
            return CRL_RESOLUTION_2048_1088_256;
        }
        return CRL_RESOLUTION_NONE;
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

        throw std::runtime_error("failed to find supported m_Format!");
    }

    inline VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
        return findSupportedFormat(physicalDevice,
                                   {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }


    // Create an m_Image memory barrier for changing the layout of
    // an m_Image and put it into an active command buffer
    // See chapter 11.4 "Image Layout" for details

    inline void setImageLayout(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkImageSubresourceRange subresourceRange,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask) {
        // Create an m_Image barrier object
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
                // Make sure any reads from the m_Image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                // Image is a transfer destination
                // Make sure any writes to the m_Image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                // Image is read by a shader
                // Make sure any shader reads from the m_Image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;
            default:
                // Other source layouts aren't firstUpdate (yet)
                break;
        }

        // Target layouts (new)
        // Destination access mask controls the dependency for the new m_Image layout
        switch (newImageLayout) {
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                // Image will be used as a transfer destination
                // Make sure any writes to the m_Image have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                // Image will be used as a transfer source
                // Make sure any reads from the m_Image have been finished
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
                // Image will be read in a shader (m_Sampler, input attachment)
                // Make sure any writes to the m_Image have been finished
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


    inline void loadShader(const char *fileName, const VkDevice &device, VkShaderModule *module) {
        std::ifstream is(fileName, std::ios::binary | std::ios::ate);

        if (is.is_open()) {
            std::streamsize size = is.tellg();
            is.seekg(0, std::ios::beg);
            std::vector<char> shaderCode(size);
            is.read(shaderCode.data(), size);
            is.close();

            assert(size > 0);
            VkShaderModuleCreateInfo moduleCreateInfo{};
            moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleCreateInfo.codeSize = size;
            moduleCreateInfo.pCode = (uint32_t *) shaderCode.data();
            VkResult res = vkCreateShaderModule(device, &moduleCreateInfo, nullptr, module);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create shader module");
        } else {
            Log::Logger::getInstance()->info("Failed to open shader file {}", fileName);
        }
    }


    template<typename T>
    size_t getIndexOf(const std::vector<T> &vecOfElements, const T &element) {
        // Find given element in vector
        auto it = std::find(vecOfElements.begin(), vecOfElements.end(), element);
        if (it != vecOfElements.end())
            return std::distance(vecOfElements.begin(), it);
        else
            return 0;
    }


};

#endif //MULTISENSE_VIEWER_UTILS_H
