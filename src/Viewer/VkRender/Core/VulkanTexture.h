//
// Created by mgjer on 24/08/2024.
//

#ifndef MULTISENSE_VIEWER_VULKANTEXTURE_H
#define MULTISENSE_VIEWER_VULKANTEXTURE_H


#include "Viewer/VkRender/Core/Texture.h"
#include "Viewer/VkRender/Core/VulkanImage.h"

namespace VkRender {
    struct VulkanTexture2DCreateInfo {

        explicit VulkanTexture2DCreateInfo(VulkanDevice& device) : vulkanDevice(device){

        }
        VulkanDevice &vulkanDevice;
        std::shared_ptr<VulkanImage> image;

    };

    class VulkanTexture {
    public:
        explicit VulkanTexture(VulkanTexture2DCreateInfo &createInfo);

        ~VulkanTexture();
        VkDescriptorImageInfo &getDescriptorInfo() { return m_imageInfo; }

    protected:
        VulkanDevice &m_vulkanDevice;
        VkSampler m_sampler;
        std::shared_ptr<VulkanImage> m_image;
        VkDescriptorImageInfo m_imageInfo = {};

    };

    class VulkanTexture2D : public VulkanTexture {
    public:
        explicit VulkanTexture2D(VulkanTexture2DCreateInfo &createInfo) : VulkanTexture(createInfo){

        }

        void loadImage(void *data, uint32_t size);
    };
}


#endif //MULTISENSE_VIEWER_VULKANTEXTURE_H
