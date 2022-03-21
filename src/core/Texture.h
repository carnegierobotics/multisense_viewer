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

#include <ktx.h>

#include <ktxvulkan.h>

#include "Buffer.h"
#include "VulkanDevice.h"
#include "tiny_gltf.h"


class Texture {
public:
    VulkanDevice *device;
    VkImage image;
    VkImageLayout imageLayout;
    VkDeviceMemory deviceMemory;
    VkImageView view;
    uint32_t width, height;
    uint32_t mipLevels;
    uint32_t layerCount;
    VkDescriptorImageInfo descriptor;
    VkSampler sampler;
    VkSamplerYcbcrConversion YUVSamplerToRGB;

    struct TextureSampler {
        VkFilter magFilter;
        VkFilter minFilter;
        VkSamplerAddressMode addressModeU;
        VkSamplerAddressMode addressModeV;
        VkSamplerAddressMode addressModeW;
    };

    void updateDescriptor();
    void destroy() const;

    ktxResult loadKTXFile(std::string filename, ktxTexture **target);
    // Load a texture from a glTF image (stored as vector of chars loaded via stb_image) and generate a full mip chaing for it

};

class Texture2D : public Texture {
public:
    void loadFromFile(
            std::string filename,
            VkFormat format,
            VulkanDevice *device,
            VkQueue copyQueue,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            bool forceLinear = false);

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

    void fromglTfImage(tinygltf::Image& gltfimage, TextureSampler textureSampler, VulkanDevice* device, VkQueue copyQueue);

    void updateTexture(void *buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight,
                       VulkanDevice *device, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags,
                       VkImageLayout imageLayout);
};

class Texture2DArray : public Texture {
public:
    void loadFromFile(
            std::string filename,
            VkFormat format,
            VulkanDevice *device,
            VkQueue copyQueue,
            VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
            VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
};

class TextureCubeMap : public Texture {
public:
    void loadFromFile(std::string filename, VulkanDevice *device, VkQueue copyQueue,
                      VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout);
};


class TextureVideo : public Texture {

public:
    VkDeviceSize bufferSize;

    TextureVideo() = default;
    TextureVideo(uint32_t texWidth, uint32_t texHeight, VkDeviceSize bufferSize, VulkanDevice *device,
                 VkImageLayout layout, VkFormat format);
    void updateTextureFromBuffer(void *buffer);

    void updateTextureFromBufferYUV(void *chromaBuffer, uint32_t chromaBufferSize, void *lumaBuffer,
                                    uint32_t lumaBufferSize);
};

#endif //MULTISENSE_TEXTURE_H