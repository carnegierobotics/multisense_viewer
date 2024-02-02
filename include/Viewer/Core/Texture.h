/**
 * @file: MultiSense-Viewer/include/Viewer/Core/Texture.h
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

#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <tiny_gltf.h>
#include <filesystem>

#include "Viewer/Core/Buffer.h"
#include "Viewer/Core/VulkanDevice.h"

class Texture {
public:
    VulkanDevice *m_Device = nullptr;
    VkImage m_Image{};
    VkImageLayout m_ImageLayout{};
    VkDeviceMemory m_DeviceMemory{};
    VkImageView m_View{};
    uint32_t m_Width = 0, m_Height = 0, m_Depth = 0;
    uint32_t m_MipLevels = 0;
    uint32_t m_LayerCount = 0;
    VkDescriptorImageInfo m_Descriptor{};
    VkSampler m_Sampler{};
    VkSamplerYcbcrConversion m_YUVSamplerToRGB{};
    VkFormat m_Format{};
    VkImageType m_Type{};
    VkImageViewType m_ViewType{};


    struct TextureSampler {
        VkFilter magFilter;
        VkFilter minFilter;
        VkSamplerAddressMode addressModeU;
        VkSamplerAddressMode addressModeV;
        VkSamplerAddressMode addressModeW;
    };


    Texture() = default;

    void updateDescriptor();


    ~Texture() {

        if (m_Width != 0 && m_Height != 0) {
            vkDestroyImageView(m_Device->m_LogicalDevice, m_View, nullptr);
            vkDestroyImage(m_Device->m_LogicalDevice, m_Image, nullptr);
            if (m_Sampler) {
                vkDestroySampler(m_Device->m_LogicalDevice, m_Sampler, nullptr);
            }
            vkFreeMemory(m_Device->m_LogicalDevice, m_DeviceMemory, nullptr);
        }
    }
};


class Texture2D : public Texture {
public:
    Texture2D() = default;

    ~Texture2D() = default;

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


    explicit Texture2D(VulkanDevice *const pDevice) {
        m_Device = pDevice;
    }

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

    void fromKtxFile(const std::string& filename, VkFormat format, VulkanDevice *device, VkQueue copyQueue,
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

    ~TextureVideo() {

        switch (m_Format) {
            case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
            case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
                vkUnmapMemory(m_Device->m_LogicalDevice, m_TexMemSecondary);
                vkFreeMemory(m_Device->m_LogicalDevice, m_TexMemSecondary, nullptr);
                vkDestroyBuffer(m_Device->m_LogicalDevice, m_TexBufferSecondary, nullptr);
                vkDestroySamplerYcbcrConversion(m_Device->m_LogicalDevice, m_YUVSamplerToRGB, nullptr);
                vkUnmapMemory(m_Device->m_LogicalDevice, m_TexMem);
                vkFreeMemory(m_Device->m_LogicalDevice, m_TexMem, nullptr);
                vkDestroyBuffer(m_Device->m_LogicalDevice, m_TexBuffer, nullptr);
                break;
            default:
                vkUnmapMemory(m_Device->m_LogicalDevice, m_TexMem);
                vkFreeMemory(m_Device->m_LogicalDevice, m_TexMem, nullptr);
                vkDestroyBuffer(m_Device->m_LogicalDevice, m_TexBuffer, nullptr);
        }
    }

    TextureVideo(uint32_t texWidth, uint32_t texHeight, VulkanDevice *device, VkImageLayout layout,
                 VkFormat format, VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT, bool sharedQueue = false);

    VkSamplerYcbcrConversionInfo createYUV420Sampler(VkFormat format);

    void createDefaultSampler();

    void updateTextureFromBuffer(VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    void updateTextureFromBufferYUV();

    //void updateTextureFromBufferYUV(VkRender::MP4Frame *frame);

};

class TextureCubeMap : public Texture {
public:
    TextureCubeMap() = default;

    void loadFromFile(const std::filesystem::path &path, VulkanDevice *device,
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


class TextureCuda : public Texture {
public:

    TextureCuda() = default;

    void fromBuffer(
            void *buffer,
            VkDeviceSize bufferSize,
            VkFormat format,
            uint32_t texWidth,
            uint32_t texHeight,
            VulkanDevice *device,
            VkQueue copyQueue,
            uint32_t* memSize,
            VkFilter filter = VK_FILTER_LINEAR,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            bool sharedQueue = false);


};
#endif //MULTISENSE_TEXTURE_H