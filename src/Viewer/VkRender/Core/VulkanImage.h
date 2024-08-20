//
// Created by mgjer on 20/08/2024.
//

#ifndef MULTISENSE_VIEWER_VULKANIMAGE_H
#define MULTISENSE_VIEWER_VULKANIMAGE_H

#include "Viewer/VkRender/Core/VulkanDevice.h"
#include "vk_mem_alloc.h"

namespace VkRender {
    struct VulkanImageCreateInfo {
        VulkanImageCreateInfo() = delete;

        VulkanImageCreateInfo(VulkanDevice &device, VmaAllocator &alloc)
                : vulkanDevice(device), allocator(alloc) {
        }
        VulkanImageCreateInfo(VulkanDevice &device, VmaAllocator &alloc,  VkImageCreateInfo& imageCI,
        VkImageViewCreateInfo& viewCI)
                : vulkanDevice(device), allocator(alloc), imageCreateInfo(imageCI), imageViewCreateInfo(viewCI) {
        }


        VulkanDevice &vulkanDevice;
        VmaAllocator &allocator;
        VkImageCreateInfo imageCreateInfo{};
        VkImageViewCreateInfo imageViewCreateInfo{};
        bool setLayout = false;
        VkImageLayout srcLayout{};
        VkImageLayout dstLayout{};
        std::string debugInfo = "Unnamed";
        VkImageAspectFlags aspectMask{};
    };

    struct VulkanImage {
    public:

        VulkanImage() = delete;

        explicit VulkanImage(VulkanImageCreateInfo &createInfo);

        // Implement move constructor
        VulkanImage(VulkanImage &&other) noexcept: m_vulkanDevice(other.m_vulkanDevice),
                                                   m_allocator(other.m_allocator) {
            std::swap(this->m_image, other.m_image);
            std::swap(this->m_mem, other.m_mem);
            std::swap(this->m_view, other.m_view);
            std::swap(this->m_allocation, other.m_allocation);
        }

        // and move assignment operator
        VulkanImage &operator=(VulkanImage &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                std::swap(this->m_vulkanDevice, other.m_vulkanDevice);
                std::swap(this->m_image, other.m_image);
                std::swap(this->m_mem, other.m_mem);
                std::swap(this->m_view, other.m_view);
                std::swap(this->m_allocation, other.m_allocation);
            }
            return *this;
        }

        // No copying allowed
        VulkanImage(const VulkanImage &) = delete;

        VulkanImage &operator=(const VulkanImage &) = delete;

        ~VulkanImage();

        VkImageView& view(){return m_view;}
        VkImage& image(){return m_image;}
        uint32_t getImageSizeRBGA(){return m_imageSize;}

    private:
        VulkanDevice &m_vulkanDevice;
        VmaAllocator &m_allocator;

        VkImage m_image{};
        VkDeviceMemory m_mem{};
        VkImageView m_view{};
        VmaAllocation m_allocation{};
        uint32_t m_imageSize;
    };
};


#endif //MULTISENSE_VIEWER_VULKANIMAGE_H
