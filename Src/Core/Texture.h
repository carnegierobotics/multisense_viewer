//
// Created by magnus on 9/13/21.
//

#ifndef MULTISENSE_TEXTURE_H
#define MULTISENSE_TEXTURE_H

/*
* Vulkan texture loader
*
* Copyright(C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"

#include "Buffer.h"
#include "VulkanDevice.h"
#include "tiny_gltf.h"

#include "Definitions.h"


class Texture {
public:
    VulkanDevice *m_Device = nullptr;
    VkImage m_Image{};
    VkImageLayout m_ImageLayout{};
    VkDeviceMemory m_DeviceMemory{};
    VkImageView m_View{};
    uint32_t m_Width = 0, m_Height = 0;
    uint32_t m_MipLevels = 0;
    uint32_t m_LayerCount = 0;
    VkDescriptorImageInfo m_Descriptor{};
    VkSampler m_Sampler{};
    VkSamplerYcbcrConversion m_YUVSamplerToRGB{};
    VkFormat m_Format{};

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
    //ktxResult loadKTXFile(std::string filename, ktxTexture **target);
    // Load a texture from a glTF m_Image (stored as vector of chars loaded via stb_image) and generate a full mip chaing for it

};


class Texture2D : public Texture {
public:
    Texture2D() = default;

    ~Texture2D() {

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
            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


    void
    fromglTfImage(tinygltf::Image &gltfimage, TextureSampler textureSampler, VulkanDevice *device, VkQueue copyQueue);
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
                 VkFormat format);

    VkSamplerYcbcrConversionInfo createYUV420Sampler(VkFormat format);

    void createDefaultSampler();

    void updateTextureFromBuffer();

    void updateTextureFromBufferYUV();

    //void updateTextureFromBufferYUV(VkRender::MP4Frame *frame);

};

#endif //MULTISENSE_TEXTURE_H