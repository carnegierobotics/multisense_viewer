//
// Created by magnus on 9/13/21.
// Based of Sascha Willems vulkan texture loading
// Copyright(C) by Sascha Willems - www.saschawillems.de
// This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)


#ifdef WIN32
#define NOMINMAX
#endif

#include <MultiSense/src/Tools/Utils.h>
#include <MultiSense/src/Tools/Macros.h>
#include "Texture.h"


void Texture::updateDescriptor() {
    descriptor.sampler = sampler;
    descriptor.imageView = view;
    descriptor.imageLayout = imageLayout;
}

void Texture2D::fromglTfImage(tinygltf::Image &gltfimage, TextureSampler textureSampler, VulkanDevice *device,
                              VkQueue copyQueue) {
    this->device = device;

    unsigned char *buffer = nullptr;
    VkDeviceSize bufferSize = 0;
    bool deleteBuffer = false;
    if (gltfimage.component == 3) {
        // Most devices don't support RGB only on Vulkan so convert if necessary
        // TODO: Check actual format support and transform only if required
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

    VkFormatProperties formatProperties;

    width = gltfimage.width;
    height = gltfimage.height;
    mipLevels = static_cast<uint32_t>(floor(log2(std::max(width, height))) + 1.0);
    mipLevels = 2;
    vkGetPhysicalDeviceFormatProperties(device->physicalDevice, format, &formatProperties);
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
    CHECK_RESULT(vkCreateBuffer(device->logicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));
    vkGetBufferMemoryRequirements(device->logicalDevice, stagingBuffer, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &memAllocInfo, nullptr, &stagingMemory));
    CHECK_RESULT(vkBindBufferMemory(device->logicalDevice, stagingBuffer, stagingMemory, 0));

    uint8_t* data = nullptr;
    CHECK_RESULT(vkMapMemory(device->logicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));
    memcpy(data, buffer, bufferSize);
    vkUnmapMemory(device->logicalDevice, stagingMemory);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {width, height, 1};
    imageCreateInfo.usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    CHECK_RESULT(vkCreateImage(device->logicalDevice, &imageCreateInfo, nullptr, &image));
    vkGetImageMemoryRequirements(device->logicalDevice, image, &memReqs);
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits,
                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &memAllocInfo, nullptr, &deviceMemory));
    CHECK_RESULT(vkBindImageMemory(device->logicalDevice, image, deviceMemory, 0));

    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.levelCount = 1;
    subresourceRange.layerCount = 1;

    Utils::setImageLayout(copyCmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          subresourceRange, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    Utils::copyBufferToImage(copyCmd, stagingBuffer, image, width, height, VK_IMAGE_ASPECT_COLOR_BIT);

    Utils::setImageLayout(copyCmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    device->flushCommandBuffer(copyCmd, copyQueue, true);

    vkFreeMemory(device->logicalDevice, stagingMemory, nullptr);
    vkDestroyBuffer(device->logicalDevice, stagingBuffer, nullptr);

    // Generate the mip chain (glTF uses jpg and png, so we need to create this manually)
    VkCommandBuffer blitCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    for (uint32_t i = 1; i < mipLevels; i++) {
        VkImageBlit imageBlit{};

        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcSubresource.mipLevel = i - 1;
        imageBlit.srcOffsets[1].x = int32_t(width >> (i - 1));
        imageBlit.srcOffsets[1].y = int32_t(height >> (i - 1));
        imageBlit.srcOffsets[1].z = 1;

        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstSubresource.mipLevel = i;
        imageBlit.dstOffsets[1].x = int32_t(width >> i);
        imageBlit.dstOffsets[1].y = int32_t(height >> i);
        imageBlit.dstOffsets[1].z = 1;

        VkImageSubresourceRange mipSubRange = {};
        mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mipSubRange.baseMipLevel = i;
        mipSubRange.levelCount = 1;
        mipSubRange.layerCount = 1;

        Utils::setImageLayout(copyCmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              mipSubRange, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);


        vkCmdBlitImage(blitCmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlit, VK_FILTER_LINEAR);


        Utils::setImageLayout(blitCmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mipSubRange, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    subresourceRange.levelCount = mipLevels;

    Utils::setImageLayout(blitCmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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
    samplerInfo.maxLod = (float) mipLevels;
    samplerInfo.maxAnisotropy = 8.0f;
    CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerInfo, nullptr, &sampler));

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                           VK_COMPONENT_SWIZZLE_A};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.levelCount = mipLevels;
    CHECK_RESULT(vkCreateImageView(device->logicalDevice, &viewInfo, nullptr, &view));

    descriptor.sampler = sampler;
    descriptor.imageView = view;
    descriptor.imageLayout = imageLayout;
    if (deleteBuffer)
        delete[] buffer;
}

void
Texture2D::fromBuffer(void *buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight,
                      VulkanDevice *device, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags,
                      VkImageLayout imageLayout) {
    assert(buffer);

    this->device = device;
    width = texWidth;
    height = texHeight;
    mipLevels = 1;

    // Create a host-visible staging buffer that contains the raw image data
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
    vkGetBufferMemoryRequirements(device->logicalDevice, stagingBuffer, &memReqs);
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    memAlloc.allocationSize = memReqs.size;


    // Copy texture data into staging buffer
    uint8_t *data = nullptr;
    CHECK_RESULT(vkMapMemory(device->logicalDevice, stagingMemory, 0, memReqs.size, 0, (void **) &data));
    memcpy(data, buffer, bufferSize);
    vkUnmapMemory(device->logicalDevice, stagingMemory);

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    // Create optimal tiled target image
    VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {width, height, 1};
    imageCreateInfo.usage = imageUsageFlags;
    // Ensure that the TRANSFER_DST bit is set for staging
    if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    CHECK_RESULT(vkCreateImage(device->logicalDevice, &imageCreateInfo, nullptr, &image));

    vkGetImageMemoryRequirements(device->logicalDevice, image, &memReqs);

    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &memAlloc, nullptr, &deviceMemory));

    CHECK_RESULT(vkBindImageMemory(device->logicalDevice, image, deviceMemory, 0));

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = mipLevels;
    subresourceRange.layerCount = 1;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );

    // Change texture image layout to shader read after all mip levels have been copied
    this->imageLayout = imageLayout;
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            imageLayout,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    device->flushCommandBuffer(copyCmd, copyQueue);

    // Clean up staging resources
    vkFreeMemory(device->logicalDevice, stagingMemory, nullptr);
    vkDestroyBuffer(device->logicalDevice, stagingBuffer, nullptr);

    // Create sampler
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

    CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerCreateInfo, nullptr, &sampler));

    // Create image view
    VkImageViewCreateInfo viewCreateInfo = {};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.pNext = nullptr;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = format;
    viewCreateInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                 VK_COMPONENT_SWIZZLE_A};
    viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.image = image;

    CHECK_RESULT(vkCreateImageView(device->logicalDevice, &viewCreateInfo, nullptr, &view));

    // Update descriptor image info member that can be used for setting up descriptor sets
    updateDescriptor();
}

Texture2D::Texture2D(void *buffer, VkDeviceSize bufferSize, VkFormat format, uint32_t texWidth, uint32_t texHeight,
                     VulkanDevice *device, VkQueue copyQueue, VkFilter filter, VkImageUsageFlags imageUsageFlags,
                     VkImageLayout imageLayout) {

    fromBuffer(buffer, bufferSize, format, texWidth, texHeight, device, copyQueue, filter, imageUsageFlags, imageLayout);
}

/**
 * @details before render loop:
1. Create your VkImage with UNDEFINED

1st to Nth frame (aka render loop):
Transition image to GENERAL
Synchronize (likely with VkFence)
Map the image, write it, unmap it (well, the mapping and unmaping can perhaps be outside render loop)
Synchronize (potentially done implicitly)
Transition image to whatever layout you need next
Do your rendering and whatnot
start over at 1.
*/
TextureVideo::TextureVideo(uint32_t texWidth, uint32_t texHeight, VulkanDevice *device, VkImageLayout layout,
                           VkFormat format) : Texture() {

    this->device = device;
    width = texWidth;
    height = texHeight;
    mipLevels = 1;
    this->format = format;

    // Create optimal tiled target image
    VkImageCreateInfo imageCreateInfo = Populate::imageCreateInfo();
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = format;
    imageCreateInfo.mipLevels = mipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.extent = {width, height, 1};
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    // Ensure that the TRANSFER_DST bit is set for staging
    if (!(imageCreateInfo.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    CHECK_RESULT(vkCreateImage(device->logicalDevice, &imageCreateInfo, nullptr, &image));
    VkMemoryAllocateInfo memAlloc = Populate::memoryAllocateInfo();
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device->logicalDevice, image, &memReqs);

    memAlloc.allocationSize = memReqs.size;
    memAlloc.memoryTypeIndex = device->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    CHECK_RESULT(vkAllocateMemory(device->logicalDevice, &memAlloc, nullptr, &deviceMemory));

    CHECK_RESULT(vkBindImageMemory(device->logicalDevice, image, deviceMemory, 0));

    VkImageViewCreateInfo viewCreateInfo = {};
    VkSamplerYcbcrConversionInfo samplerYcbcrConversionInfo{};

    // Create sampler dependt on image format
    switch (format) {
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R16_UINT:
            // Create grayscale texture image
            createDefaultSampler();
            viewCreateInfo.pNext = nullptr;
            break;
        case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
        case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
            samplerYcbcrConversionInfo = createYUV420Sampler(format);
            viewCreateInfo.pNext = &samplerYcbcrConversionInfo;

            // Create YUV sampler
            break;
        default:
            std::cerr << "No texture sampler for that format yet\n";
            break;
    }


    // Create image view
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = format;
    viewCreateInfo.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B};
    viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.image = image;

    CHECK_RESULT(vkCreateImageView(device->logicalDevice, &viewCreateInfo, nullptr, &view));

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = mipLevels;
    subresourceRange.layerCount = 1;
    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Change texture image layout to shader read after all mip levels have been copied
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            layout,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    this->imageLayout = layout;

    device->flushCommandBuffer(copyCmd, device->transferQueue);

    // Update descriptor image info member that can be used for setting up descriptor sets
    updateDescriptor();

    // Create empty buffers we can copy our texture data to



    // Create sampler dependt on image format
    switch (format) {
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_UINT:
            size = (VkDeviceSize) width * height * 2;
        CHECK_RESULT(device->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                size,
                &stagingBuffer,
                &stagingMemory));

            CHECK_RESULT(vkMapMemory(device->logicalDevice, stagingMemory, 0, size, 0, (void **) &data));
            break;
        case VK_FORMAT_R8_UNORM:
            size = (VkDeviceSize) width * height;
            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    size,
                    &stagingBuffer,
                    &stagingMemory));
            CHECK_RESULT(vkMapMemory(device->logicalDevice, stagingMemory, 0, size, 0, (void **) &data));
            break;
        case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
        case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
            size = (VkDeviceSize) width * height;

            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    size,
                    &stagingBuffer,
                    &stagingMemory));
            CHECK_RESULT(vkMapMemory(device->logicalDevice, stagingMemory, 0, size, 0, (void **) &data));

            size = (VkDeviceSize) width * height;

            CHECK_RESULT(device->createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    size,
                    &stagingBuffer2,
                    &stagingMemory2));
            CHECK_RESULT(vkMapMemory(device->logicalDevice, stagingMemory2, 0, size, 0, (void **) &data2));
            break;
        default:
            std::cerr << "No video texture type for that format yet\n";
            break;
    }
}


void TextureVideo::updateTextureFromBuffer() {

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = mipLevels;
    subresourceRange.layerCount = 1;

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy mip levels from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );

    // Change texture image layout to shader read after all mip levels have been copied
    this->imageLayout = imageLayout;
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    device->flushCommandBuffer(copyCmd, device->transferQueue);
}


void TextureVideo::updateTextureFromBufferYUV() {
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = mipLevels;
    subresourceRange.layerCount = 1;

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    VkBufferImageCopy bufferCopyRegionChroma = {};
    bufferCopyRegionChroma.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
    bufferCopyRegionChroma.imageSubresource.mipLevel = 0;
    bufferCopyRegionChroma.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegionChroma.imageSubresource.layerCount = 1;
    bufferCopyRegionChroma.imageExtent.width = width / 2;
    bufferCopyRegionChroma.imageExtent.height = height / 2;
    bufferCopyRegionChroma.imageExtent.depth = 1;
    bufferCopyRegionChroma.bufferOffset = 0;

    // Use a separate command buffer for texture loading
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy data from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );


    // Copy data from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            stagingBuffer2,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegionChroma
    );

    // Change texture image layout to shader read after all mip levels have been copied
    this->imageLayout = imageLayout;
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    device->flushCommandBuffer(copyCmd, device->transferQueue);


}
/*
void TextureVideo::updateTextureFromBufferYUV(VkRender::MP4Frame *frame) {


    // Create a host-visible staging buffer that contains the raw image data
    VkBuffer plane0, plane1, plane2;
    VkDeviceMemory planeMem0, planeMem1, planeMem2;

    CHECK_RESULT(device->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame->plane0Size,
            &plane0,
            &planeMem0, frame->plane0));

    CHECK_RESULT(device->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame->plane1Size,
            &plane1,
            &planeMem1, frame->plane1));

    CHECK_RESULT(device->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame->plane2Size,
            &plane2,
            &planeMem2, frame->plane2));


    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = mipLevels;
    subresourceRange.layerCount = 1;

    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
    bufferCopyRegion.imageSubresource.mipLevel = 0;
    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = width;
    bufferCopyRegion.imageExtent.height = height;
    bufferCopyRegion.imageExtent.depth = 1;
    bufferCopyRegion.bufferOffset = 0;

    VkBufferImageCopy bufferCopyRegionPlane1 = {};
    bufferCopyRegionPlane1.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
    bufferCopyRegionPlane1.imageSubresource.mipLevel = 0;
    bufferCopyRegionPlane1.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegionPlane1.imageSubresource.layerCount = 1;
    bufferCopyRegionPlane1.imageExtent.width = width / 2;
    bufferCopyRegionPlane1.imageExtent.height = height / 2;
    bufferCopyRegionPlane1.imageExtent.depth = 1;
    bufferCopyRegionPlane1.bufferOffset = 0;

    VkBufferImageCopy bufferCopyRegionPlane2 = {};
    bufferCopyRegionPlane2.imageSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_2_BIT;
    bufferCopyRegionPlane2.imageSubresource.mipLevel = 0;
    bufferCopyRegionPlane2.imageSubresource.baseArrayLayer = 0;
    bufferCopyRegionPlane2.imageSubresource.layerCount = 1;
    bufferCopyRegionPlane2.imageExtent.width = width / 2;
    bufferCopyRegionPlane2.imageExtent.height = height / 2;
    bufferCopyRegionPlane2.imageExtent.depth = 1;
    bufferCopyRegionPlane2.bufferOffset = 0;

    // Use a separate command buffer for texture loading|
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy data from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            plane0,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegion
    );

    // Copy data from staging buffer
    vkCmdCopyBufferToImage(
            copyCmd,
            plane1,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegionPlane1
    );

    vkCmdCopyBufferToImage(
            copyCmd,
            plane2,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &bufferCopyRegionPlane2
    );

    // Change texture image layout to shader read after all mip levels have been copied
    this->imageLayout = imageLayout;
    Utils::setImageLayout(
            copyCmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresourceRange,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    device->flushCommandBuffer(copyCmd, device->transferQueue);


    vkFreeMemory(device->logicalDevice, planeMem0, nullptr);
    vkDestroyBuffer(device->logicalDevice, plane0, nullptr);

    vkFreeMemory(device->logicalDevice, planeMem1, nullptr);
    vkDestroyBuffer(device->logicalDevice, plane1, nullptr);

    vkFreeMemory(device->logicalDevice, planeMem2, nullptr);
    vkDestroyBuffer(device->logicalDevice, plane2, nullptr);

}
*/
VkSamplerYcbcrConversionInfo TextureVideo::createYUV420Sampler(VkFormat format) {

    // YUV TEXTURE SAMPLER
    VkSamplerYcbcrConversionCreateInfo info = {VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO};

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
    info.chromaFilter = VK_FILTER_LINEAR;

    // COSITED or MIDPOINT? I think normal YUV420p content is MIDPOINT,
    // but not quite sure ...
    info.xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;
    info.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT;

    // Not sure what this is for.
    info.forceExplicitReconstruction = VK_FALSE;

    // For YUV420p.
    info.format = format;

    vkCreateSamplerYcbcrConversion(device->logicalDevice, &info, nullptr,
                                   &YUVSamplerToRGB);


    VkSamplerYcbcrConversionInfo samplerConversionInfo{VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO, nullptr,
                                                       YUVSamplerToRGB};
// Create sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.pNext = &samplerConversionInfo;

    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.maxAnisotropy = VK_FALSE;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;


    CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerCreateInfo, nullptr, &sampler));

    VkSamplerYcbcrConversionInfo samplerYcbcrConversionInfo{};
    samplerYcbcrConversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO;
    samplerYcbcrConversionInfo.conversion = YUVSamplerToRGB;
    samplerYcbcrConversionInfo.pNext = VK_NULL_HANDLE;

    return samplerYcbcrConversionInfo;

}

void TextureVideo::createDefaultSampler() {

    // Create sampler
    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.maxAnisotropy = 1.0f;

    CHECK_RESULT(vkCreateSampler(device->logicalDevice, &samplerCreateInfo, nullptr, &sampler));

}

