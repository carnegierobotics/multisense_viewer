/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/Utils.h
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
 *   2021-09-4, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_UTILS_H
#define MULTISENSE_VIEWER_UTILS_H

#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>
#include <vulkan/vulkan_core.h>
#include <regex>
#include <algorithm>

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <direct.h>
#include <windows.h>
#include <UserEnv.h>

#else

#include <sys/stat.h>
#include <tiffio.h>

#endif

#include "Viewer/Tools/Macros.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Tools/Logger.h"
#include "MultiSense/MultiSenseTypes.hh"

namespace Utils {
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FUNCTION

    static std::filesystem::path getShadersPath() {
        return {"./Assets/Shaders"};
    }

    static std::filesystem::path getAssetsPath() {
        return {"./Assets/"};
    }

    static std::filesystem::path getTexturePath() {
        return {"./Assets/Textures"};
    }

    static std::filesystem::path getModelsPath() {
        return {"./Assets/Models"};
    }

    static std::filesystem::path getScriptsPath() {
        return {"Scripts/"};
    }

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

    static VkRender::CRLCameraDataType CRLSourceToTextureType(const std::string &d) {
        if (d == "Luma Left") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Right") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Rectified Left") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Rectified Right") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Left") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Right") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Rectified Left") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Compressed Rectified Right") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Disparity Left") return VkRender::CRL_DISPARITY_IMAGE;

        if (d == "Color Aux") return VkRender::CRL_COLOR_IMAGE_YUV420;
        if (d == "Color Rectified Aux") return VkRender::CRL_COLOR_IMAGE_YUV420;
        if (d == "Luma Aux") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Luma Rectified Aux") return VkRender::CRL_GRAYSCALE_IMAGE;
        if (d == "Compressed Aux") return VkRender::CRL_COLOR_IMAGE_YUV420;
        if (d == "Compressed Rectified Aux") return VkRender::CRL_COLOR_IMAGE_YUV420;

        return VkRender::CRL_CAMERA_IMAGE_NONE;
    }

    DISABLE_WARNING_POP


    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool isInVector(const std::vector<std::string> &v, const std::string &str) {
        if (std::find(v.begin(), v.end(), str) != v.end())
            return true;
        return false;

    }

    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool isStreamRunning(VkRender::Device &dev, const std::string &stream) {
        // Check if disparity stream is running
        for (auto &ch: dev.channelInfo) {
            if (ch.state != VkRender::CRL_STATE_ACTIVE)
                continue;

            if (Utils::isInVector(ch.enabledStreams, stream)) {
                return true;
            }
        }
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


    inline VkRender::CRLCameraResolution stringToCameraResolution(const std::string &resolution) {
        if (resolution == "960 x 600 x 64x") return VkRender::CRL_RESOLUTION_960_600_64;
        if (resolution == "960 x 600 x 128x") return VkRender::CRL_RESOLUTION_960_600_128;
        if (resolution == "960 x 600 x 256x") return VkRender::CRL_RESOLUTION_960_600_256;
        if (resolution == "1920 x 1200 x 64x") return VkRender::CRL_RESOLUTION_1920_1200_64;
        if (resolution == "1920 x 1200 x 128x") return VkRender::CRL_RESOLUTION_1920_1200_128;
        if (resolution == "1920 x 1200 x 256x") return VkRender::CRL_RESOLUTION_1920_1200_256;
        if (resolution == "1024 x 1024 x 128x") return VkRender::CRL_RESOLUTION_1024_1024_128;
        if (resolution == "2048 x 1088 x 256x") return VkRender::CRL_RESOLUTION_2048_1088_256;
        return VkRender::CRL_RESOLUTION_NONE;
    }

    inline std::string cameraResolutionToString(const VkRender::CRLCameraResolution &res) {
        switch (res) {
            case VkRender::CRL_RESOLUTION_960_600_64:
                return "960 x 600 x 64";
            case VkRender::CRL_RESOLUTION_960_600_128:
                return "960 x 600 x 128";
            case VkRender::CRL_RESOLUTION_960_600_256:
                return "960 x 600 x 256";
            case VkRender::CRL_RESOLUTION_1920_1200_64:
                return "1920 x 1200 x 64";
            case VkRender::CRL_RESOLUTION_1920_1200_128:
                return "1920 x 1200 x 128";
            case VkRender::CRL_RESOLUTION_1920_1200_256:
                return "1920 x 1200 x 256";
            case VkRender::CRL_RESOLUTION_1024_1024_128:
                return "1024 x 1024 x 128";
            case VkRender::CRL_RESOLUTION_2048_1088_256:
                return "2048 x 1088 x 256";
            case VkRender::CRL_RESOLUTION_NONE:
                return "Resolution not supported";
        }
        return "Resolution not supported";
    }

    /** @brief Convert camera resolution enum to uint32_t values used by the libmultisense */
    inline void
    cameraResolutionToValue(VkRender::CRLCameraResolution resolution, uint32_t *_width, uint32_t *_height,
                            uint32_t *_depth) {
        uint32_t width = 0, height = 0, depth = 0;
        switch (resolution) {
            case VkRender::CRL_RESOLUTION_NONE:
                width = 0;
                height = 0;
                depth = 0;
                break;
            case VkRender::CRL_RESOLUTION_960_600_64:
                width = 960;
                height = 600;
                depth = 64;
                break;
            case VkRender::CRL_RESOLUTION_960_600_128:
                width = 960;
                height = 600;
                depth = 128;
                break;
            case VkRender::CRL_RESOLUTION_960_600_256:
                width = 960;
                height = 600;
                depth = 256;
                break;
            case VkRender::CRL_RESOLUTION_1920_1200_64:
                width = 1920;
                height = 1200;
                depth = 64;
                break;
            case VkRender::CRL_RESOLUTION_1920_1200_128:
                width = 1920;
                height = 1200;
                depth = 128;
                break;
            case VkRender::CRL_RESOLUTION_1920_1200_256:
                width = 1920;
                height = 1200;
                depth = 256;
                break;
            case VkRender::CRL_RESOLUTION_1024_1024_128:
                width = 1024;
                height = 1024;
                depth = 128;
                break;
            case VkRender::CRL_RESOLUTION_2048_1088_256:
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
    inline VkRender::CRLCameraResolution
    valueToCameraResolution(uint32_t _width, uint32_t _height, uint32_t _depth) {
        if (_height == 600 && _width == 960 && _depth == 64) {
            return VkRender::CRL_RESOLUTION_960_600_64;
        }
        if (_height == 600 && _width == 960 && _depth == 128) {
            return VkRender::CRL_RESOLUTION_960_600_128;
        }
        if (_height == 600 && _width == 960 && _depth == 256) {
            return VkRender::CRL_RESOLUTION_960_600_256;
        }
        if (_height == 1200 && _width == 1920 && _depth == 64) {
            return VkRender::CRL_RESOLUTION_1920_1200_64;
        }
        if (_height == 1200 && _width == 1920 && _depth == 128) {
            return VkRender::CRL_RESOLUTION_1920_1200_128;
        }
        if (_height == 1200 && _width == 1920 && _depth == 256) {
            return VkRender::CRL_RESOLUTION_1920_1200_256;
        }
        if (_height == 1024 && _width == 1024 && _depth == 128) {
            return VkRender::CRL_RESOLUTION_1024_1024_128;
        }
        if (_height == 2048 && _width == 1088 && _depth == 256) {
            return VkRender::CRL_RESOLUTION_2048_1088_256;
        }
        return VkRender::CRL_RESOLUTION_NONE;
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
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                break;
            default:
                // Other source layouts aren't firstUpdate (yet)
                Log::Logger::getInstance()->error("Source layouts arent updated yet. No Image transition completed");
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
                Log::Logger::getInstance()->error("Source layouts arent updated yet. No Image transition completed");
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
            moduleCreateInfo.
                    sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleCreateInfo.
                    codeSize = size;
            moduleCreateInfo.
                    pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            VkResult res = vkCreateShaderModule(device, &moduleCreateInfo, nullptr,
                                                module);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create shader module");
        } else {
            Log::Logger::getInstance()->error("Failed to open shader file {}", fileName);
        }
    }

    inline VkPipelineShaderStageCreateInfo
    loadShader(VkDevice device, std::string fileName, VkShaderStageFlagBits stage, VkShaderModule *module) {
        // Check if we have .spv extensions. If not then add it.
        std::size_t extension = fileName.find(".spv");
        if (extension == std::string::npos)
            fileName.append(".spv");
        Utils::loadShader((Utils::getShadersPath().append(fileName)).string().c_str(),
                          device, module);
        assert(module != VK_NULL_HANDLE);

        VkPipelineShaderStageCreateInfo shaderStage = {};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = stage;
        shaderStage.module = *module;
        shaderStage.pName = "main";
        Log::Logger::getInstance()->trace("Loaded shader {} for stage {}", fileName, static_cast<uint32_t>(stage));
        return shaderStage;
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


    static inline void initializeUIDataBlockWithTestData(VkRender::Device &dev) {
        dev.state = VkRender::CRL_STATE_JUST_ADDED;
        dev.cameraName = "Simulated device";
        dev.IP = "127.0.0.1";
        dev.simulatedDevice = true;
        dev.channelInfo.resize(4); // max number of remote heads
        dev.win.clear();
        crl::multisense::RemoteHeadChannel ch = 0;
        VkRender::ChannelInfo chInfo;
        chInfo.availableSources.clear();
        chInfo.modes.clear();
        chInfo.availableSources.emplace_back("Idle");
        chInfo.index = ch;
        chInfo.state = VkRender::CRL_STATE_JUST_ADDED;
        chInfo.selectedResolutionMode = VkRender::CRL_RESOLUTION_1920_1200_128;
        std::vector<crl::multisense::system::DeviceMode> supportedDeviceModes;
        supportedDeviceModes.emplace_back();
        //initCameraModes(&chInfo.modes, supportedModes);
        chInfo.selectedResolutionMode = Utils::valueToCameraResolution(1920, 1080, 128);
        for (int i = 0; i < VkRender::CRL_PREVIEW_TOTAL_MODES; ++i) {
            dev.win[static_cast<VkRender::StreamWindowIndex>(i)].availableRemoteHeads.push_back(std::to_string(ch + 1));
            if (!chInfo.availableSources.empty())
                dev.win[static_cast<VkRender::StreamWindowIndex>(i)].selectedRemoteHeadIndex = ch;
        }

        // stop streams if there were any enabled, just so we can start with a clean slate
        //stopStreamTask(this, "All", ch);

        dev.channelInfo.at(ch) = chInfo;

        Log::Logger::getLogMetrics()->device.dev = &dev;

        // Update Debug Window
        auto &info = Log::Logger::getLogMetrics()->device.info;

        info.firmwareBuildDate = "cInfo.sensorFirmwareBuildDate";
        info.firmwareVersion = 12;
        info.apiBuildDate = "cInfo.apiBuildDate";
        info.apiVersion = 13;
        info.hardwareMagic = 14;
        info.hardwareVersion = 16;
        info.sensorFpgaDna = 17;

    }

    static inline Log::LogLevel getLogLevelEnumFromString(const std::string &logStr) {
        if (logStr == "LOG_INFO") return Log::LOG_LEVEL::LOG_LEVEL_INFO;
        else if (logStr == "LOG_TRACE") return Log::LOG_LEVEL::LOG_LEVEL_TRACE;
        else if (logStr == "LOG_DEBUG") return Log::LOG_LEVEL::LOG_LEVEL_DEBUG;
        return Log::LOG_LEVEL::LOG_LEVEL_INFO;
    }

    static inline bool stringToBool(const std::string &str) {
        std::string lowerStr;
        std::transform(str.begin(), str.end(), std::back_inserter(lowerStr),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (lowerStr == "true") {
            return true;
        } else if (lowerStr == "false") {
            return false;
        } else {
            // Handle invalid input string, e.g., throw an exception or return a default value
            throw std::invalid_argument("Invalid input string for boolean conversion");
        }
    }

    static inline std::string boolToString(bool value) {
        return value ? "true" : "false";
    }


/**
 * Returns the systemCache path for Windows/Ubuntu. If it doesn't exist it is created
 * @return path to cache folder
 */
    static inline std::filesystem::path getSystemCachePath() {
        // ON windows this file should be places in the user cache folder //
#ifdef WIN32
        const char *envVarName = "APPDATA";
        char *envVarValue = nullptr;
        size_t envVarValueSize = 0;
        _dupenv_s(&envVarValue, &envVarValueSize, envVarName);

        std::filesystem::path cachePath = envVarValue;
        std::filesystem::path multiSenseCachePath = cachePath / "multisense";
#else
        std::filesystem::path cachePath = std::getenv("HOME");
        cachePath /= ".cache";
        std::filesystem::path multiSenseCachePath = cachePath / "multisense";
#endif

        if (!std::filesystem::exists(multiSenseCachePath)) {
            std::error_code ec;
            if (std::filesystem::create_directories(multiSenseCachePath, ec)) {
                Log::Logger::getInstance((multiSenseCachePath / "logger.log").string())->info(
                        "Created cache directory {}", multiSenseCachePath.string());
            } else {
                Log::Logger::getInstance()->error("Failed to create cache directory {}. Error Code: {}",
                                                  multiSenseCachePath.string(), ec.value());
            }
        }
        return multiSenseCachePath;
    }

    static inline std::filesystem::path getSystemHomePath() {
#ifdef WIN32
        char path[MAX_PATH];
        HANDLE hToken;
        DWORD bufferSize = sizeof(path);

        if (OpenProcessToken(GetCurrentProcess(), TOKEN_READ, &hToken)) {
            if (!GetUserProfileDirectory(hToken, path, &bufferSize)) {
                CloseHandle(hToken);
                return ""; // Failed to get home directory
            }
            CloseHandle(hToken);
        } else {
            return ""; // Failed to open process token
        }

        return std::string(path);
#else
        const char *homeDir = getenv("HOME");
        if (homeDir) {
            return std::string(homeDir);
        } else {
            return getSystemCachePath();
        }
#endif
    }

    static inline bool checkRegexMatch(const std::string &str, const std::string &expression) {
        std::string lowered_str = str;
        std::transform(str.begin(), str.end(), lowered_str.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        std::regex pattern(expression, std::regex_constants::icase);  // icase for case-insensitive matching
        return std::regex_search(lowered_str, pattern);
    }


    static inline bool isLocalVersionLess(const std::string &localVersion, const std::string &remoteVersion) {
        int localMajor, localMinor, localPatch;
        int remoteMajor, remoteMinor, remotePatch;

        // Parse local version
        std::istringstream localStream(localVersion);
        localStream >> localMajor;
        localStream.ignore(1, '.'); // Skip the dot
        localStream >> localMinor;
        localStream.ignore(1, '-'); // Skip the dot
        localStream >> localPatch;

        // Parse remote version
        std::istringstream remoteStream(remoteVersion);
        remoteStream >> remoteMajor;
        remoteStream.ignore(1, '.'); // Skip the dot
        remoteStream >> remoteMinor;
        remoteStream.ignore(1, '-'); // Skip the dot
        remoteStream >> remotePatch;


        // Compare
        if (localMajor < remoteMajor) {
            return true;
        }
        if (localMajor == remoteMajor && localMinor < remoteMinor) {
            return true;
        }
        if (localMajor == remoteMajor && localMinor == remoteMinor && localPatch < remotePatch) {
            return true;
        }
        return false;
    }


    static inline bool parseCustomMetadataToJSON(VkRender::Device *dev) {

        std::string serialNumber = Log::Logger::getLogMetrics()->device.info.serialNumber;
        nlohmann::json metadataJSON;
        uint32_t fwVersion = Log::Logger::getLogMetrics()->device.info.firmwareVersion;
        std::string fwVersionStr = fmt::format("0x{:x}", fwVersion);
        std::string firmwareBuildDate = Log::Logger::getLogMetrics()->device.info.firmwareBuildDate;
        uint32_t apiVersion = Log::Logger::getLogMetrics()->device.info.apiVersion;
        std::string apiVersionStr = fmt::format("0x{:x}", apiVersion);
        std::string apiBuildDate = Log::Logger::getLogMetrics()->device.info.apiBuildDate;

        std::filesystem::path saveFolder = dev->record.frameSaveFolder;

#ifdef WIN32
        auto t = std::time(nullptr);
        std::tm tm;
        localtime_s(&tm, &t);  // Use localtime_s instead of std::localtime
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
        auto date = oss.str();
#else
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
        auto date = oss.str();
#endif

        try {

            // Populate JSON object with metadata
            metadataJSON["metadata"]["name"] = std::string(dev->record.metadata.logName);
            metadataJSON["metadata"]["date"] = date;
            metadataJSON["metadata"]["location"] = std::string(dev->record.metadata.location);
            metadataJSON["metadata"]["description"] = std::string(dev->record.metadata.recordDescription);
            metadataJSON["metadata"]["equipment_description"] = std::string(dev->record.metadata.equipmentDescription);
            metadataJSON["metadata"]["camera_extrinsics"] =
                    (saveFolder / (serialNumber + "_extrinsics.yml")).string();
            metadataJSON["metadata"]["camera_intrinsics"] =
                    (saveFolder / (serialNumber + "_intrinsics.yml")).string();
            metadataJSON["metadata"]["camera_firmware_version"] = fwVersionStr;
            metadataJSON["metadata"]["camera_firmware_build_date"] = firmwareBuildDate;
            metadataJSON["metadata"]["camera_api_version"] = apiVersionStr;
            metadataJSON["metadata"]["camera_api_build_date"] = apiBuildDate;
            //metadataJSON["metadata"]["camera_calibration_info"] = std::string(dev->record.metadata.camera_calibration_info);
            //metadataJSON["metadata"]["camera_firmware_version"] = std::string(dev->record.metadata.camera_firmware_version);
            metadataJSON["metadata"]["tags"] = std::string(dev->record.metadata.tags);

            // Check if the user has set a camera name
            if (strlen(dev->record.metadata.camera) <= 0) {
                metadataJSON["metadata"]["camera_name"] = std::string(
                        Log::Logger::getLogMetrics()->device.dev->cameraName);
            } else {
                metadataJSON["metadata"]["camera_name"] = std::string(dev->record.metadata.camera);
            }

            // Split the string by newline characters
            std::istringstream iss(dev->record.metadata.customField);
            std::string line;
            std::map<std::string, std::string> customFields;

            while (std::getline(iss, line)) {
                // Skip empty lines
                if (line.empty()) continue;

                // Find the position of the '=' character
                size_t pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);

                    // Trim spaces
                    key.erase(0, key.find_first_not_of(' '));
                    key.erase(key.find_last_not_of(' ') + 1);
                    value.erase(0, value.find_first_not_of(' '));
                    value.erase(value.find_last_not_of(' ') + 1);

                    customFields[key] = value;
                }
            }
            metadataJSON["metadata"]["custom_fields"] = customFields;

            metadataJSON["data_info"]["session_start"] = date;

            dev->record.metadata.JSON = metadataJSON;
            dev->record.metadata.parsed = true;
        } catch (nlohmann::json::exception &e) {
            Log::Logger::getInstance()->error("Failed to parse metadata to JSON: {}", e.what());
            dev->record.metadata.parsed = false;
        }
        return dev->record.metadata.parsed;
    }

    static inline bool parseMetadataToJSON(VkRender::Device *dev) {
        std::string serialNumber = Log::Logger::getLogMetrics()->device.info.serialNumber;
        nlohmann::json metadataJSON;
        uint32_t fwVersion = Log::Logger::getLogMetrics()->device.info.firmwareVersion;
        std::string fwVersionStr = fmt::format("0x{:x}", fwVersion);
        std::string firmwareBuildDate = Log::Logger::getLogMetrics()->device.info.firmwareBuildDate;
        uint32_t apiVersion = Log::Logger::getLogMetrics()->device.info.apiVersion;
        std::string apiVersionStr = fmt::format("0x{:x}", apiVersion);
        std::string apiBuildDate = Log::Logger::getLogMetrics()->device.info.apiBuildDate;
        std::filesystem::path saveFolder = dev->record.frameSaveFolder;
        // Get todays date:
#ifdef WIN32
        auto t = std::time(nullptr);
        std::tm tm;
        localtime_s(&tm, &t);  // Use localtime_s instead of std::localtime
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
        auto date = oss.str();
#else
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
        auto date = oss.str();
#endif
        try {
            // Populate JSON object with metadata
            metadataJSON["metadata"]["date"] = date;
            metadataJSON["metadata"]["camera_extrinsics"] = (saveFolder / (serialNumber + "_extrinsics.yml")).string();
            metadataJSON["metadata"]["camera_intrinsics"] = (
                    saveFolder / (serialNumber + "_intrinsics.yml")).string();
            metadataJSON["metadata"]["camera_firmware_version"] = fwVersionStr;
            metadataJSON["metadata"]["camera_firmware_build_date"] = firmwareBuildDate;
            metadataJSON["metadata"]["camera_api_version"] = apiVersionStr;
            metadataJSON["metadata"]["camera_api_build_date"] = apiBuildDate;
            metadataJSON["metadata"]["camera_name"] = std::string(Log::Logger::getLogMetrics()->device.dev->cameraName);

            metadataJSON["data_info"]["session_start"] = date;

            dev->record.metadata.JSON = metadataJSON;
            dev->record.metadata.parsed = true;
        } catch (nlohmann::json::exception &e) {
            Log::Logger::getInstance()->error("Failed to parse metadata to JSON: {}", e.what());
            dev->record.metadata.parsed = false;
        }
        return dev->record.metadata.parsed;
    }

    static inline void writeTIFFImage(const std::filesystem::path &fileName, uint32_t width, uint32_t height, float *data) {
        int samplesPerPixel = 1;
        TIFF *out = TIFFOpen(fileName.string().c_str(), "w");
        if (!out) {
            throw std::runtime_error("Failed to open TIFF file for writing.");
        }
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);   // set number of channels per pixel
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);    // set the size of the channels
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
        //   Some other essential fields to set that you do not have to understand for now.
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

        // We set the strip size of the file to be size of one row of pixels
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, 0));
        //Now writing image to the file one strip at a time
        uint32_t row;
        for (row = 0; row < height; row++) {
            // Write each row as a strip
            if (TIFFWriteScanline(out, &data[row * width], row, 0) < 0) {
                TIFFClose(out);
                if (std::filesystem::exists(fileName)) {
                    std::filesystem::remove(fileName);
                }
            }
        }
        TIFFClose(out);
    }
}

#endif //MULTISENSE_VIEWER_UTILS_H
