/**
 * @file: MultiSense-Viewer/src/Core/Texture.cpp
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

#ifdef WIN32
#define NOMINMAX
#endif

#include <filesystem>
#include <vulkan/vulkan.h>
#include <KTX-Software/include/ktx.h>
#include <KTX-Software/include/ktxvulkan.h>

#include "Viewer/Core/Texture.h"
#include "Viewer/Tools/Macros.h"
#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"

void Texture::updateDescriptor() {
    m_Descriptor.sampler = m_Sampler;
    m_Descriptor.imageView = m_View;
    m_Descriptor.imageLayout = m_ImageLayout;
}

void Texture2D::fromglTfImage(tinygltf::Image &gltfimage, TextureSampler textureSampler, VulkanDevice *device,
                              VkQueue copyQueue) {
    this->m_Device = device;

    unsigned char *buffer = nullptr;
    VkDeviceSize bufferSize = 0;
    bool deleteBuffer = false;
    if (gltfimage.component == 3) {
        // Most devices don't support RGB only on Vulkan so convert if necessary
        bufferSize = (VkDeviceSize) gltfimage.width * gltfimage.height * 4;
        buffer = new unsigned char[bufferSize];
        unsigned char *rgba = buffer;
        unsigned char *rgb = &gltfimage.image[0];
        for (int32_t i = 0; i < gltfimage.width * gltfimage.height; ++i) {
            for (int32_t j = 0; j < 3; ++j) {
                rgba[j] = rgb[j];
            }
            rgba += 4;
            rgb += 3;
        }
        deleteBuffer = true;
    } else {
        buffer = &gltfimage.image[0];
        bufferSize = gltfimage.image.size();
    }

    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    m_Format = format;

    VkFormatProperties formatProperties;

    m_Width = gltfimage.width;
    m_Height = gltfimage.height;
    m_MipLevels = static_cast<uint32_t>(floor(log2(std::max(m_Width, m_Height))) + 1.0);
    vkGetPhysicalDeviceFormatProperties(device->m_PhysicalDevice, format, &formatProperties);
    assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT);
    assert(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);

    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkMemoryRequirements memReqs{};

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_RESULT(vkCreateBuffer(device->m_LogicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));
    vkGetBufferMemoryRequirements(device->m_LogicalDevice, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK_RESULT(vkAllocateMemory(device->m_LogicalDevice, &memAllocInfo, nullptr, &stagingMemory));
    CHECK_RESULT(vkBindBufferMemory(device->m_LogicalDevice, stagingBuffer, stagingMemory, 0));

    uint8_t *data = nullptr;
    CHECK_RESULT(vkMapMemory(device->m_LogicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));
    memcpy(data, buffer, bufferSize);
    vkUnmapMemory(device->m_LogicalDevice, stagingMemory);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = m_MipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {m_Width, m_Height, 1};
    imageCreateInfo.usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    CHECK_RESULT(vkCreateImage(device->m_LogicalDevice, &imageCreateInfo, nullptr, &m_Image));
    vkGetImageMemoryRequirements(device->m_LogicalDevice, m_Image, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_RESULT(vkAllocateMemory(device->m_LogicalDevice, &memAllocInfo, nullptr, &m_DeviceMemory));
    CHECK_RESULT(vkBindImageMemory(device->m_LogicalDevice, m_Image, m_DeviceMemory, 0));

    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.levelCount = 1;
    subresourceRange.layerCount = 1;

    Utils::setImageLayout(copyCmd, m_Image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          subresourceRange, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    Utils::copyBufferToImage(copyCmd, stagingBuffer, m_Image, m_Width, m_Height, VK_IMAGE_ASPECT_COLOR_BIT);

    Utils::setImageLayout(copyCmd, m_Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    device->flushCommandBuffer(copyCmd, copyQueue, true);

    vkFreeMemory(device->m_LogicalDevice, stagingMemory, nullptr);
    vkDestroyBuffer(device->m_LogicalDevice, stagingBuffer, nullptr);

    // Generate the mip chain (glTF uses jpg and png, so we need to create this manually)
    VkCommandBuffer blitCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    for (uint32_t i = 1; i < m_MipLevels; i++) {
        VkImageBlit imageBlit{};

        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcSubresource.mipLevel = i - 1;
        imageBlit.srcOffsets[1].x = int32_t(m_Width >> (i - 1));
        imageBlit.srcOffsets[1].y = int32_t(m_Height >> (i - 1));
        imageBlit.srcOffsets[1].z = 1;

        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstSubresource.mipLevel = i;
        imageBlit.dstOffsets[1].x = int32_t(m_Width >> i);
        imageBlit.dstOffsets[1].y = int32_t(m_Height >> i);
        imageBlit.dstOffsets[1].z = 1;

        VkImageSubresourceRange mipSubRange = {};
        mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mipSubRange.baseMipLevel = i;
        mipSubRange.levelCount = 1;
        mipSubRange.layerCount = 1;

        Utils::setImageLayout(blitCmd, m_Image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              mipSubRange, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);


        vkCmdBlitImage(blitCmd, m_Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_Image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlit, VK_FILTER_LINEAR);


        Utils::setImageLayout(blitCmd, m_Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mipSubRange, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    subresourceRange.levelCount = m_MipLevels;

    Utils::setImageLayout(blitCmd, m_Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    m_ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    device->flushCommandBuffer(blitCmd, copyQueue, true);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = textureSampler.magFilter;
    samplerInfo.minFilter = textureSampler.minFilter;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = textureSampler.addressModeU;
    samplerInfo.addressModeV = textureSampler.addressModeV;
    samplerInfo.addressModeW = textureSampler.addressModeW;
    samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.maxAnisotropy = 1.0;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxLod = (float) m_MipLevels;
    samplerInfo.maxAnisotropy = 8.0f;
    CHECK_RESULT(vkCreateSampler(device->m_LogicalDevice, &samplerInfo, nullptr, &m_Sampler));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_Image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                           VK_COMPONENT_SWIZZLE_A};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.levelCount = m_MipLevels;
    CHECK_RESULT(vkCreateImageView(device->m_LogicalDevice, &viewInfo, nullptr, &m_View));

    m_Descriptor.sampler = m_Sampler;
    m_Descriptor.imageView = m_View;
    m_Descriptor.imageLayout = m_ImageLayout;
    if (deleteBuffer)
        delete[] buffer;
}

void
Texture2D::fromBuffer(void *buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight,
                      VulkanDevice *device, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags,
                      VkImageLayout imageLayout) {
    assert(buffer);

    this->m_Device = device;
    m_Width = texWidth;
    m_Height = texHeight;
    m_MipLevels = 1;

    // Create a host-visible staging buffer that contains the raw m_Image m_DataPtr
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    CHECK_RESULT(device->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            bufferSize,
            &stagingBuffer,
            &stagingMemory, buffer));


    // Create the memory backing up the buffer handle
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device->m_LogicalDevice, stagingBuffer, &memReqs);
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    memAlloc.allocationSize = memReqs.size;


    // Copy texture m_DataPtr into staging buffer
    uint8_t *data = nullptr;
    CHECK_RESULT(vkMapMemory(device->m_LogicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));
    memcpy(data, buffer, bufferSize);
    vkUnmapMemory(device->m_LogicalDevice, stagingMemory);

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = m_Width;
    bufferCopyRegion.imageExtent.height = m_Height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    // Create optimal tiled target m_Image
    VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = m_MipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {m_Width, m_Height, 1};
    imageCreateInfo.usage = imageUsageFlags;
    // Ensure that the TRANSFER_DST bit is set for staging
    if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    CHECK_RESULT(vkCreateImage(device->m_LogicalDevice, &imageCreateInfo, nullptr, &m_Image));

    vkGetImageMemoryRequirements(device->m_LogicalDevice, m_Image, &memReqs);

    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    CHECK_RESULT(vkAllocateMemory(device->m_LogicalDevice, &memAlloc, nullptr, &m_DeviceMemory));

    CHECK_RESULT(vkBindImageMemory(device->m_LogicalDevice, m_Image, m_DeviceMemory, 0));

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = m_MipLevels;
    subresourceRange.layerCount = 1;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal m_Image (target)
    // Optimal m_Image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );

    // Change texture m_Image layout to shader read after all mip levels have been copied
    this->m_ImageLayout = imageLayout;
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            imageLayout,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    device->flushCommandBuffer(copyCmd, copyQueue);

    // Clean up staging resources
    vkFreeMemory(device->m_LogicalDevice, stagingMemory, nullptr);
    vkDestroyBuffer(device->m_LogicalDevice, stagingBuffer, nullptr);

    // Create m_Sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = filter;
    samplerCreateInfo.minFilter = filter;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.maxAnisotropy = 1.0f;

    CHECK_RESULT(vkCreateSampler(device->m_LogicalDevice, &samplerCreateInfo, nullptr, &m_Sampler));

    // Create m_Image m_View
    VkImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.pNext = nullptr;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = format;
    viewCreateInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                 VK_COMPONENT_SWIZZLE_A};
    viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.image = m_Image;

    CHECK_RESULT(vkCreateImageView(device->m_LogicalDevice, &viewCreateInfo, nullptr, &m_View));

    // Update m_Descriptor m_Image info member that can be used for setting up m_Descriptor sets
    updateDescriptor();
}

Texture2D::Texture2D(void *buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight,
                     VulkanDevice *device, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags,
                     VkImageLayout imageLayout) {

    fromBuffer(buffer, bufferSize, format, texWidth, texHeight, device, copyQueue, filter, imageUsageFlags,
               imageLayout);
}

/**
* Load a 2D texture including all mip levels
*
* @param filename File to load (supports .ktx)
* @param format Vulkan format of the image data stored in the file
* @param device Vulkan device to create the texture on
* @param copyQueue Queue used for the texture staging copy commands (must support transfer)
* @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
* @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
* @param (Optional) forceLinear Force linear tiling (not advised, defaults to false)
*
*/
void Texture2D::fromKtxFile(std::string filename, VkFormat format, VulkanDevice *device, VkQueue copyQueue,
                             VkImageUsageFlags imageUsageFlags, VkImageLayout imageLayout, bool forceLinear) {
    ktxTexture *ktxTexture;
    ktxResult result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);

    m_Device = device;
    m_Width = ktxTexture->baseWidth;
    m_Height = ktxTexture->baseHeight;
    m_MipLevels = ktxTexture->numLevels;

    ktx_uint8_t *ktxTextureData = ktxTexture_GetData(ktxTexture);
    ktx_size_t ktxTextureSize = ktxTexture_GetDataSize(ktxTexture);

    // Get device properties for the requested texture format
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(m_Device->m_PhysicalDevice, format, &formatProperties);

    // Only use linear tiling if requested (and supported by the device)
    // Support for linear tiling is mostly limited, so prefer to use
    // optimal tiling instead
    // On most implementations linear tiling will only support a very
    // limited amount of formats and features (mip maps, cubemaps, arrays, etc.)
    VkBool32 useStaging = !forceLinear;

    VkMemoryAllocateInfo memAllocInfo = Populate::memoryAllocateInfo();
    VkMemoryRequirements memReqs;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    if (useStaging) {
        // Create a host-visible staging buffer that contains the raw image data
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;

        VkBufferCreateInfo bufferCreateInfo = Populate::bufferCreateInfo();
        bufferCreateInfo.size = ktxTextureSize;
        // This buffer is used as a transfer source for the buffer copy
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;


        CHECK_RESULT(vkCreateBuffer(m_Device->m_LogicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));

        // Get memory requirements for the staging buffer (alignment, memory type bits)
        vkGetBufferMemoryRequirements(m_Device->m_LogicalDevice, stagingBuffer, &memReqs);

        memAllocInfo.allocationSize = memReqs.size;
        // Get memory type index for a host visible buffer
        memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);


        CHECK_RESULT(vkAllocateMemory(m_Device->m_LogicalDevice, &memAllocInfo, nullptr, &stagingMemory));

        CHECK_RESULT(vkBindBufferMemory(m_Device->m_LogicalDevice, stagingBuffer, stagingMemory, 0));

        // Copy texture data into staging buffer
        uint8_t *data;

        CHECK_RESULT(vkMapMemory(m_Device->m_LogicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));
        memcpy(data, ktxTextureData, ktxTextureSize);
        vkUnmapMemory(m_Device->m_LogicalDevice, stagingMemory);

        // Setup buffer copy regions for each mip level
        std::vector<VkBufferImageCopy> bufferCopyRegions;

        for (uint32_t i = 0; i < m_MipLevels; i++) {
            ktx_size_t offset;
            KTX_error_code result = ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
            assert(result == KTX_SUCCESS);

            VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = i;
            bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = std::max(1u, ktxTexture->baseWidth >> i);
            bufferCopyRegion.imageExtent.height = std::max(1u, ktxTexture->baseHeight >> i);
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = offset;

            bufferCopyRegions.push_back(bufferCopyRegion);
        }

        // Create optimal tiled target image
        VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = format;
        imageCreateInfo.mipLevels = m_MipLevels;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.extent = {m_Width, m_Height, 1};
        imageCreateInfo.usage = imageUsageFlags;
        // Ensure that the TRANSFER_DST bit is set for staging
        if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        }

        CHECK_RESULT(vkCreateImage(m_Device->m_LogicalDevice, &imageCreateInfo, nullptr, &m_Image));

        vkGetImageMemoryRequirements(m_Device->m_LogicalDevice, m_Image, &memReqs);

        memAllocInfo.allocationSize = memReqs.size;

        memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        CHECK_RESULT(vkAllocateMemory(m_Device->m_LogicalDevice, &memAllocInfo, nullptr, &m_DeviceMemory));

        CHECK_RESULT(vkBindImageMemory(m_Device->m_LogicalDevice, m_Image, m_DeviceMemory, 0));

        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = m_MipLevels;
        subresourceRange.layerCount = 1;

        // Image barrier for optimal image (target)
        // Optimal image will be used as destination for the copy
        Utils::setImageLayout(
                copyCmd,
                m_Image,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresourceRange,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Copy mip levels from staging buffer
        vkCmdCopyBufferToImage(
                copyCmd,
                stagingBuffer,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<uint32_t>(bufferCopyRegions.size()),
                bufferCopyRegions.data()
        );

        // Change texture image layout to shader read after all mip levels have been copied
        m_ImageLayout = imageLayout;
        Utils::setImageLayout(
                copyCmd,
                m_Image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                imageLayout,
                subresourceRange,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        device->flushCommandBuffer(copyCmd, copyQueue);

        // Clean up staging resources
        vkFreeMemory(m_Device->m_LogicalDevice, stagingMemory, nullptr);
        vkDestroyBuffer(m_Device->m_LogicalDevice, stagingBuffer, nullptr);
    } else {
        // Prefer using optimal tiling, as linear tiling
        // may support only a small set of features
        // depending on implementation (e.g. no mip maps, only one layer, etc.)

        // Check if this support is supported for linear tiling
        assert(formatProperties.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);

        VkImage mappableImage;
        VkDeviceMemory mappableMemory;

        VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent = {m_Width, m_Height, 1};
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imageCreateInfo.usage = imageUsageFlags;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // Load mip map level 0 to linear tiling image

        CHECK_RESULT(vkCreateImage(m_Device->m_LogicalDevice, &imageCreateInfo, nullptr, &mappableImage));

        // Get memory requirements for this image
        // like size and alignment
        vkGetImageMemoryRequirements(m_Device->m_LogicalDevice, mappableImage, &memReqs);
        // Set memory allocation size to required memory size
        memAllocInfo.allocationSize = memReqs.size;

        // Get memory type that can be mapped to host memory
        memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        // Allocate host memory

        CHECK_RESULT(vkAllocateMemory(m_Device->m_LogicalDevice, &memAllocInfo, nullptr, &mappableMemory));

        // Bind allocated image for use

        CHECK_RESULT(vkBindImageMemory(m_Device->m_LogicalDevice, mappableImage, mappableMemory, 0));

        // Get sub resource layout
        // Mip map count, array layer, etc.
        VkImageSubresource subRes = {};
        subRes.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subRes.mipLevel = 0;

        VkSubresourceLayout subResLayout;
        void *data;

        // Get sub resources layout
        // Includes row pitch, size offsets, etc.
        vkGetImageSubresourceLayout(m_Device->m_LogicalDevice, mappableImage, &subRes, &subResLayout);

        // Map image memory

        CHECK_RESULT(vkMapMemory(m_Device->m_LogicalDevice, mappableMemory, 0, memReqs.size, 0, &data));

        // Copy image data into memory
        memcpy(data, ktxTextureData, memReqs.size);

        vkUnmapMemory(m_Device->m_LogicalDevice, mappableMemory);

        // Linear tiled images don't need to be staged
        // and can be directly used as textures
        m_Image = mappableImage;
        m_DeviceMemory = mappableMemory;
        m_ImageLayout = imageLayout;

        // Setup image memory barrier
        Utils::setImageLayout(copyCmd, m_Image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, imageLayout,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        device->flushCommandBuffer(copyCmd, copyQueue);
    }

    ktxTexture_Destroy(ktxTexture);

    // Create a default sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    // Max level-of-detail should match mip level count
    samplerCreateInfo.maxLod = (useStaging) ? (float) m_MipLevels : 0.0f;
    // Only enable anisotropic filtering if enabled on the device
    samplerCreateInfo.maxAnisotropy = m_Device->m_EnabledFeatures.samplerAnisotropy
                                      ? m_Device->m_Properties.limits.maxSamplerAnisotropy : 1.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    CHECK_RESULT(vkCreateSampler(m_Device->m_LogicalDevice, &samplerCreateInfo, nullptr, &m_Sampler));

    // Create image view
    // Textures are not directly accessed by the shaders and
    // are abstracted by image views containing additional
    // information and sub resource ranges
    VkImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = format;
    viewCreateInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                 VK_COMPONENT_SWIZZLE_A};
    viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Linear tiling usually won't support mip maps
    // Only set mip map count if optimal tiling is used
    viewCreateInfo.subresourceRange.levelCount = (useStaging) ? m_MipLevels : 1;
    viewCreateInfo.image = m_Image;

    CHECK_RESULT(vkCreateImageView(m_Device->m_LogicalDevice, &viewCreateInfo, nullptr, &m_View));

    // Update descriptor image info member that can be used for setting up descriptor sets
    updateDescriptor();
}


/**
 * @details before render loop:
1. Create your VkImage with UNDEFINED

1st to Nth frame (aka render loop):
Transition m_Image to GENERAL
Synchronize (likely with VkFence)
Map the m_Image, write it, unmap it (well, the mapping and unmaping can perhaps be outside render loop)
Synchronize (potentially done implicitly)
Transition m_Image to whatever layout you need next
Do your rendering and whatnot
start over at 1.
*/
TextureVideo::TextureVideo(uint32_t texWidth, uint32_t texHeight, VulkanDevice *device, VkImageLayout layout,
                           VkFormat format) : Texture() {

    assert(texWidth != 0 && texHeight != 0);

    this->m_Device = device;
    m_Width = texWidth;
    m_Height = texHeight;
    m_MipLevels = 1;
    this->m_Format = format;

    // Create optimal tiled target m_Image
    VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = m_MipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {m_Width, m_Height, 1};
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    // Ensure that the TRANSFER_DST bit is set for staging
    if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    CHECK_RESULT(vkCreateImage(device->m_LogicalDevice, &imageCreateInfo, nullptr, &m_Image));
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device->m_LogicalDevice, m_Image, &memReqs);

    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    CHECK_RESULT(vkAllocateMemory(device->m_LogicalDevice, &memAlloc, nullptr, &m_DeviceMemory));

    CHECK_RESULT(vkBindImageMemory(device->m_LogicalDevice, m_Image, m_DeviceMemory, 0));

    VkImageViewCreateInfo viewCreateInfo = {};
    VkSamplerYcbcrConversionInfo samplerYcbcrConversionInfo{};

    // Create m_Sampler dependt on m_Image m_Format
    switch (format) {
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R16_UINT:
            // Create grayscale texture m_Image
            createDefaultSampler();
            viewCreateInfo.pNext = nullptr;
            break;
        case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
        case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
            samplerYcbcrConversionInfo = createYUV420Sampler(format);
            viewCreateInfo.pNext = &samplerYcbcrConversionInfo;

            // Create YUV m_Sampler
            break;

        case VK_FORMAT_R8G8B8A8_UNORM:
            createDefaultSampler();
            break;
        default:
            std::cerr << "No texture m_Sampler for that m_Format yet\n";
            break;
    }


    // Create m_Image m_View
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = format;
    viewCreateInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                 VK_COMPONENT_SWIZZLE_A};
    viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.image = m_Image;

    CHECK_RESULT(vkCreateImageView(device->m_LogicalDevice, &viewCreateInfo, nullptr, &m_View));

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = m_MipLevels;
    subresourceRange.layerCount = 1;
    // Image barrier for optimal m_Image (target)
    // Optimal m_Image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Change texture m_Image layout to shader read after all mip levels have been copied
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            layout,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    this->m_ImageLayout = layout;

    device->flushCommandBuffer(copyCmd, device->m_TransferQueue);

    // Update m_Descriptor m_Image info member that can be used for setting up m_Descriptor sets
    updateDescriptor();

    // Create empty buffers we can copy our texture m_DataPtr to

    // Create m_Sampler dependt on m_Image m_Format
    switch (format) {
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_UINT:
            m_TexSize = (VkDeviceSize) m_Width * m_Height * 2;
            break;
        case VK_FORMAT_R8_UNORM:
            m_TexSize = (VkDeviceSize) m_Width * m_Height;
            break;
        case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
        case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
            m_TexSize = (VkDeviceSize) m_Width * m_Height;
            m_TexSizeSecondary = (VkDeviceSize) m_Width * m_Height / 2;
            break;
        case VK_FORMAT_R8G8B8A8_UNORM:
            m_TexSize = (VkDeviceSize) m_Width * m_Height * 4;
            break;
        default:
            std::cerr << "No video texture type for that m_Format yet\n";
            break;
    }

    CHECK_RESULT(device->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            m_TexSize,
            &m_TexBuffer,
            &m_TexMem));

    CHECK_RESULT(vkMapMemory(device->m_LogicalDevice, m_TexMem, 0, m_TexSize, 0, (void **) &m_DataPtr));

    Log::Logger::getInstance()->info("Allocated Texture GPU memory {} bytes with format {}", m_TexSize, (int) format);

    if (m_TexSizeSecondary != 0) {
        CHECK_RESULT(device->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_TexSizeSecondary,
                &m_TexBufferSecondary,
                &m_TexMemSecondary));
        CHECK_RESULT(vkMapMemory(device->m_LogicalDevice, m_TexMemSecondary, 0, m_TexSizeSecondary, 0,
                                 (void **) &m_DataPtrSecondary));
        Log::Logger::getInstance()->info("Allocated Secondary Texture GPU memory {} bytes with format {}",
                                         m_TexSizeSecondary, (int) format);

    }


}


void TextureVideo::updateTextureFromBuffer() {
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = m_MipLevels;
    subresourceRange.layerCount = 1;

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = m_Width;
    bufferCopyRegion.imageExtent.height = m_Height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = m_Device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal m_Image (target)
    // Optimal m_Image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            m_TexBuffer,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );

    // Change texture m_Image layout to shader read after all mip levels have been copied
    this->m_ImageLayout = m_ImageLayout;
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    m_Device->flushCommandBuffer(copyCmd, m_Device->m_TransferQueue);
}


void TextureVideo::updateTextureFromBufferYUV() {
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = m_MipLevels;
    subresourceRange.layerCount = 1;

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = m_Width;
    bufferCopyRegion.imageExtent.height = m_Height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    VkBufferImageCopy bufferCopyRegionChroma = {};
    bufferCopyRegionChroma.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
    bufferCopyRegionChroma.imageSubresource.mipLevel = 0;
    bufferCopyRegionChroma.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegionChroma.imageSubresource.layerCount = 1;
    bufferCopyRegionChroma.imageExtent.width = m_Width / 2;
    bufferCopyRegionChroma.imageExtent.height = m_Height / 2;
    bufferCopyRegionChroma.imageExtent.depth = 1;
    bufferCopyRegionChroma.bufferOffset = 0;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = m_Device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal m_Image (target)
    // Optimal m_Image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy m_DataPtr from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            m_TexBuffer,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );


    // Copy m_DataPtr from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            m_TexBufferSecondary,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegionChroma
    );

    // Change texture m_Image layout to shader read after all mip levels have been copied
    this->m_ImageLayout = m_ImageLayout;
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    m_Device->flushCommandBuffer(copyCmd, m_Device->m_TransferQueue);


}

VkSamplerYcbcrConversionInfo TextureVideo::createYUV420Sampler(VkFormat format) {

    // YUV TEXTURE SAMPLER
    VkSamplerYcbcrConversionCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO;

    // Which 3x3 YUV to RGB matrix is used?
    // 601 is generally used for SD content.
    // 709 for HD content.
    // 2020 for UHD content.
    // Can also use IDENTITY which lets you sample the raw YUV and
    // do the conversion in shader code.
    // At least you don't have to hit the texture unit 3 times.
    info.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;

    // TV (NARROW) or PC (FULL) range for YUV?
    // Usually, JPEG uses full range and broadcast content is narrow.
    // If using narrow, the YUV components need to be
    // rescaled before it can be converted.
    info.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;

    // Deal with order of components.
    info.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
    };

    // With NEAREST, chroma is duplicated to a 2x2 block for YUV420p.
    // In fancy video players, you might even get bicubic/sinc
    // interpolation filters for chroma because why not ...
    info.chromaFilter = VK_FILTER_NEAREST;

    // COSITED or MIDPOINT? I think normal YUV420p content is MIDPOINT,
    // but not quite sure ...
    info.xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
    info.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;

    // Not sure what this is for.
    info.forceExplicitReconstruction = VK_FALSE;

    // For YUV420p.
    info.format = format;

    vkCreateSamplerYcbcrConversion(m_Device->m_LogicalDevice, &info, nullptr,
                                   &m_YUVSamplerToRGB);


    VkSamplerYcbcrConversionInfo samplerConversionInfo{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO, nullptr,
                                                       m_YUVSamplerToRGB};
    // Create m_Sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.pNext = &samplerConversionInfo;

    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.maxAnisotropy = VK_FALSE;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;


    CHECK_RESULT(vkCreateSampler(m_Device->m_LogicalDevice, &samplerCreateInfo, nullptr, &m_Sampler));

    VkSamplerYcbcrConversionInfo samplerYcbcrConversionInfo{};
    samplerYcbcrConversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
    samplerYcbcrConversionInfo.conversion = m_YUVSamplerToRGB;
    samplerYcbcrConversionInfo.pNext = VK_NULL_HANDLE;

    return samplerYcbcrConversionInfo;

}

void TextureVideo::createDefaultSampler() {

    // Create m_Sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.maxAnisotropy = 1.0f;

    CHECK_RESULT(vkCreateSampler(m_Device->m_LogicalDevice, &samplerCreateInfo, nullptr, &m_Sampler));
}

void TextureCubeMap::loadFromFile(const std::filesystem::path &path,
                                  VulkanDevice *device,
                                  VkImageUsageFlags imageUsageFlags,
                                  VkImageLayout imageLayout) {
    m_Device = device;
    ktxTexture *ktxTexture;

    ktxResult result = KTX_SUCCESS;
    result = ktxTexture_CreateFromNamedFile(path.string().c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
    assert(result == KTX_SUCCESS);
    Log::Logger::getInstance()->info("Loading TextureCubeMap {}", reinterpret_cast<const char*>(path.c_str()));
    m_Width = ktxTexture->baseWidth;
    m_Height = ktxTexture->baseHeight;
    m_MipLevels = ktxTexture->numLevels;

    ktx_uint8_t *ktxTextureData = ktxTexture_GetData(ktxTexture);
    ktx_size_t ktxTextureSize = ktxTexture_GetDataSize(ktxTexture);
    VkFormat format = ktxTexture_GetVkFormat(ktxTexture);
    VkMemoryAllocateInfo memAllocInfo = Populate::memoryAllocateInfo();
    VkMemoryRequirements memReqs;
    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    VkBufferCreateInfo bufferCreateInfo = Populate::bufferCreateInfo();
    bufferCreateInfo.size = ktxTextureSize;
    // This buffer is used as a transfer source for the buffer copy
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;


    CHECK_RESULT(vkCreateBuffer(device->m_LogicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));
    // Get memory requirements for the staging buffer (alignment, memory type bits)
    vkGetBufferMemoryRequirements(device->m_LogicalDevice, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    // Get memory type index for a host visible buffer
    memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    CHECK_RESULT(vkAllocateMemory(device->m_LogicalDevice, &memAllocInfo, nullptr, &stagingMemory));
    CHECK_RESULT(vkBindBufferMemory(device->m_LogicalDevice, stagingBuffer, stagingMemory, 0));

    // Copy texture data into staging buffer
    uint8_t *data;

    CHECK_RESULT(vkMapMemory(device->m_LogicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));
    memcpy(data, ktxTextureData, ktxTextureSize);
    vkUnmapMemory(device->m_LogicalDevice, stagingMemory);

    // Setup buffer copy regions for each face including all of its mip levels
    std::vector<VkBufferImageCopy> bufferCopyRegions;

    for (uint32_t face = 0; face < 6; face++) {
        for (uint32_t level = 0; level < m_MipLevels; level++) {
            ktx_size_t offset;
            KTX_error_code result = ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
            assert(result == KTX_SUCCESS);

            VkBufferImageCopy bufferCopyRegion = {};
            bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            bufferCopyRegion.imageSubresource.mipLevel = level;
            bufferCopyRegion.imageSubresource.baseArrayLayer = face;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> level;
            bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> level;
            bufferCopyRegion.imageExtent.depth = 1;
            bufferCopyRegion.bufferOffset = offset;

            bufferCopyRegions.push_back(bufferCopyRegion);
        }
    }

    // Create optimal tiled target image
    VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = m_MipLevels;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {m_Width, m_Height, 1};
    imageCreateInfo.usage = imageUsageFlags | VK_IMAGE_USAGE_SAMPLED_BIT;
    // Ensure that the TRANSFER_DST bit is set for staging
    if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    // Cube faces count as array layers in Vulkan
    imageCreateInfo.arrayLayers = 6;
    // This flag is required for Cube map images
    imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;


    CHECK_RESULT(vkCreateImage(device->m_LogicalDevice, &imageCreateInfo, nullptr, &m_Image));

    vkGetImageMemoryRequirements(device->m_LogicalDevice, m_Image, &memReqs);

    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);


    CHECK_RESULT(vkAllocateMemory(device->m_LogicalDevice, &memAllocInfo, nullptr, &m_DeviceMemory));

    CHECK_RESULT(vkBindImageMemory(device->m_LogicalDevice, m_Image, m_DeviceMemory, 0));

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal image (target)
    // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = m_MipLevels;
    subresourceRange.layerCount = 6;

    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy the Cube map faces from the staging buffer to the optimal tiled image
    vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(bufferCopyRegions.size()),
            bufferCopyRegions.data());

    // Change texture image layout to shader read after all faces have been copied
    this->m_ImageLayout = imageLayout;
    Utils::setImageLayout(
            copyCmd,
            m_Image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            imageLayout,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    device->flushCommandBuffer(copyCmd, device->m_TransferQueue);

    // Create sampler
    VkSamplerCreateInfo samplerCreateInfo = Populate::samplerCreateInfo();
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
    samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.maxAnisotropy = device->m_EnabledFeatures.samplerAnisotropy
                                      ? device->m_Properties.limits.maxSamplerAnisotropy : 1.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = (float) m_MipLevels;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

    (vkCreateSampler(device->m_LogicalDevice, &samplerCreateInfo, nullptr, &m_Sampler));

    // Create image view
    VkImageViewCreateInfo viewCreateInfo = Populate::imageViewCreateInfo();
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewCreateInfo.format = format;
    viewCreateInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                 VK_COMPONENT_SWIZZLE_A};
    viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    viewCreateInfo.subresourceRange.layerCount = 6;
    viewCreateInfo.subresourceRange.levelCount = m_MipLevels;
    viewCreateInfo.image = m_Image;

    CHECK_RESULT(vkCreateImageView(device->m_LogicalDevice, &viewCreateInfo, nullptr, &m_View));

    // Clean up staging resources
    ktxTexture_Destroy(ktxTexture);
    vkFreeMemory(device->m_LogicalDevice, stagingMemory, nullptr);
    vkDestroyBuffer(device->m_LogicalDevice, stagingBuffer, nullptr);

    // Update descriptor image info member that can be used for setting up descriptor sets
    updateDescriptor();

}
