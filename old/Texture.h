/**
 * @file: MultiSense-Viewer/include/Viewer/VkRender/Core/Texture.h
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
 *   2021-09-13, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_TEXTURE_H
#define MULTISENSE_TEXTURE_H

#pragma once

#include "Viewer/Application/pch.h"

#include <tiny_gltf.h>

#include "Viewer/Rendering/Core/Buffer.h"
#include "Viewer/Rendering/Core/VulkanDevice.h"
#include "Viewer/Tools/Macros.h"
#include "VulkanResourceManager.h"

class Texture {
public:
    VulkanDevice *m_device = nullptr;
    VkImage m_image = VK_NULL_HANDLE;
    VkImageLayout m_imageLayout{};
    VkDeviceMemory m_deviceMemory = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
    uint32_t m_width = 0, m_height = 0, m_depth = 0;
    uint32_t m_mipLevels = 0;
    uint32_t m_layerCount = 0;
    VkDescriptorImageInfo m_descriptor{};
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkSamplerYcbcrConversion m_YUVSamplerToRGB = VK_NULL_HANDLE;
    VkFormat m_format{};
    VkImageType m_type{};
    VkImageViewType m_viewType{};


    struct TextureSampler {
        VkFilter magFilter;
        VkFilter minFilter;
        VkSamplerAddressMode addressModeU;
        VkSamplerAddressMode addressModeV;
        VkSamplerAddressMode addressModeW;
    };


    // TODO Somehow exchange is not available in llvm sycl branch?
    template<class T, class U = T>
    T exchange(T &obj, U &&new_value) {
        T old_value = std::move(obj);
        obj = std::forward<U>(new_value);
        return old_value;
    }

    Texture() = default;

    virtual ~Texture() {

        if (m_device != nullptr) {


            auto logicalDevice = m_device->m_LogicalDevice;
            VkFence fence;
            VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
            vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence);

            VkImageView view = m_view;
            VkImage image = m_image;
            VkSampler sampler = m_sampler;
            VkDeviceMemory memory = m_deviceMemory;


            VkRender::VulkanResourceManager::getInstance().deferDeletion(
                    [logicalDevice, view, image, sampler, memory]() {
                        vkDestroyImageView(logicalDevice, view, nullptr);
                        vkDestroyImage(logicalDevice, image, nullptr);
                        if (sampler) {
                            vkDestroySampler(logicalDevice, sampler, nullptr);
                        }
                        vkFreeMemory(logicalDevice, memory, nullptr);
                    },
                    fence);
        }
    }

    void updateDescriptor();

};


class Texture2D : public Texture {
public:
    Texture2D() = default;


    // Move constructor
    Texture2D(Texture2D &&other) noexcept: Texture(std::move(other)) {
        // Move constructor logic specific to Texture2D
        m_device = exchange(other.m_device, nullptr);
        m_image = exchange(other.m_image, {});
        m_deviceMemory = exchange(other.m_deviceMemory, {});
        m_view = exchange(other.m_view, {});
        m_sampler = exchange(other.m_sampler, {});
        m_YUVSamplerToRGB = exchange(other.m_YUVSamplerToRGB, {});
        m_imageLayout = other.m_imageLayout;
        m_width = other.m_width;
        m_height = other.m_height;
        m_depth = other.m_depth;
        m_mipLevels = other.m_mipLevels;
        m_layerCount = other.m_layerCount;
        m_descriptor = other.m_descriptor;
        m_format = other.m_format;
        m_type = other.m_type;
        m_viewType = other.m_viewType;
    }

    // Move assignment operator
    Texture2D &operator=(Texture2D &&other) noexcept {
        if (this != &other) {
            // Proper cleanup of existing resources
            // Move all resources from 'other' to 'this'
            m_device = exchange(other.m_device, nullptr);
            m_image = exchange(other.m_image, {});
            m_deviceMemory = exchange(other.m_deviceMemory, {});
            m_view = exchange(other.m_view, {});
            m_sampler = exchange(other.m_sampler, {});
            m_YUVSamplerToRGB = exchange(other.m_YUVSamplerToRGB, {});
            // Copy simple types
            m_imageLayout = other.m_imageLayout;
            m_width = other.m_width;
            m_height = other.m_height;
            m_depth = other.m_depth;
            m_mipLevels = other.m_mipLevels;
            m_layerCount = other.m_layerCount;
            m_descriptor = other.m_descriptor;
            m_format = other.m_format;
            m_type = other.m_type;
            m_viewType = other.m_viewType;
        }
        return *this;
    }


    Texture2D(void *buffer,
              VkDeviceSize bufferSize,
              VkFormat format,
              uint32_t texWidth,
              uint32_t texHeight,
              VulkanDevice *device,
              VkQueue copyQueue,
              VkFilter filter = VK_FILTER_LINEAR,
              VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
              VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    void fromBuffer(
            void *buffer,
            VkDeviceSize bufferSize,
            VkFormat format,
            uint32_t texWidth,
            uint32_t texHeight,
            VulkanDevice *device,
            VkQueue copyQueue,
            VkFilter filter = VK_FILTER_LINEAR,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            bool sharedQueue = false);


    void
    fromglTfImage(tinygltf::Image &gltfimage, TextureSampler textureSampler, VulkanDevice *device, VkQueue copyQueue);

    void fromKtxFile(const std::string &filename, VkFormat format, VulkanDevice *device, VkQueue copyQueue,
                     VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
                     VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, bool forceLinear = false);
};


class TextureVideo : public Texture {

public:
    // Create a host-visible staging buffer that contains the raw m_Image m_DataPtr
    VkBuffer m_TexBuffer{};
    VkDeviceMemory m_TexMem{};
    VkDeviceSize m_TexSize = 0;
    uint8_t *m_DataPtr{};

    VkBuffer m_TexBufferSecondary{};
    VkDeviceMemory m_TexMemSecondary{};
    VkDeviceSize m_TexSizeSecondary = 0;
    uint8_t *m_DataPtrSecondary{};

    TextureVideo() = default;

    ~TextureVideo() override {
        if (m_device) {


            switch (m_format) {
                case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
                case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
                    vkUnmapMemory(m_device->m_LogicalDevice, m_TexMemSecondary);
                    vkFreeMemory(m_device->m_LogicalDevice, m_TexMemSecondary, nullptr);
                    vkDestroyBuffer(m_device->m_LogicalDevice, m_TexBufferSecondary, nullptr);
                    vkDestroySamplerYcbcrConversion(m_device->m_LogicalDevice, m_YUVSamplerToRGB, nullptr);
                    vkUnmapMemory(m_device->m_LogicalDevice, m_TexMem);
                    vkFreeMemory(m_device->m_LogicalDevice, m_TexMem, nullptr);
                    vkDestroyBuffer(m_device->m_LogicalDevice, m_TexBuffer, nullptr);
                    break;
                default:
                    vkUnmapMemory(m_device->m_LogicalDevice, m_TexMem);
                    vkFreeMemory(m_device->m_LogicalDevice, m_TexMem, nullptr);
                    vkDestroyBuffer(m_device->m_LogicalDevice, m_TexBuffer, nullptr);
            }
        }
    }

    TextureVideo(uint32_t texWidth, uint32_t texHeight, VulkanDevice *device, VkImageLayout layout,
                 VkFormat format, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT, bool sharedQueue = false);

    VkSamplerYcbcrConversionInfo createYUV420Sampler(VkFormat format);

    void createDefaultSampler();

    void updateTextureFromBuffer(void *data, uint32_t size, VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                 VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    void updateTextureFromBufferYUV();

    //void updateTextureFromBufferYUV(VkRender::MP4Frame *frame);

};

class TextureCubeMap : public Texture {
public:
    TextureCubeMap() = default;

    // Move constructor
    TextureCubeMap(TextureCubeMap &&other) noexcept: Texture(std::move(other)) {
        // Move constructor logic specific to TextureCubeMap
    }

    // Move assignment operator
    TextureCubeMap &operator=(TextureCubeMap &&other) noexcept {
        if (this != &other) {
            // Proper cleanup of existing resources
            // Move all resources from 'other' to 'this'
            m_device = exchange(other.m_device, nullptr);
            m_image = exchange(other.m_image, {});
            m_deviceMemory = exchange(other.m_deviceMemory, {});
            m_view = exchange(other.m_view, {});
            m_sampler = exchange(other.m_sampler, {});
            m_YUVSamplerToRGB = exchange(other.m_YUVSamplerToRGB, {});
            // Copy simple types
            m_imageLayout = other.m_imageLayout;
            m_width = other.m_width;
            m_height = other.m_height;
            m_depth = other.m_depth;
            m_mipLevels = other.m_mipLevels;
            m_layerCount = other.m_layerCount;
            m_descriptor = other.m_descriptor;
            m_format = other.m_format;
            m_type = other.m_type;
            m_viewType = other.m_viewType;
        }
        return *this;
    }

    void fromKtxFile(const std::filesystem::path &path, VulkanDevice *device,
                     VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
                     VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
};

class Texture3D : public Texture {
public:
    Texture3D() = default;

    void fromBuffer(
            void *buffer,
            VkDeviceSize bufferSize,
            VkFormat format,
            uint32_t texWidth,
            uint32_t texHeight,
            uint32_t texDepth,
            VulkanDevice *device,
            VkQueue copyQueue,
            VkFilter filter = VK_FILTER_LINEAR,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            bool sharedQueue = false);
};

#endif //MULTISENSE_TEXTURE_H