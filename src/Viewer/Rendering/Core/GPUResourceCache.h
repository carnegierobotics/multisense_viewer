//
// Created by mgjer on 27/04/2025.
//

#ifndef GPURESOURCECACHE_H
#define GPURESOURCECACHE_H


#include "Viewer/Rendering/Core/ShaderModuleCache.h"
#include "Viewer/Rendering/Core/TextureCache.h"
#include "Viewer/Rendering/Core/ImageCache.h"

namespace VkRender {

class GPUResourceCache {
public:
  explicit GPUResourceCache(VulkanDevice& device)
    : shaderModules(device.m_LogicalDevice)
    , textures(device.m_LogicalDevice)
    , images(device.m_LogicalDevice)
  {}

  ShaderModuleCache      shaderModules;
  TextureCache textures;
  VulkanImageCache images;


  // ... other GPU‐side caches (framebuffers, samplers, etc.) ...
};

} // namespace VkRender

#endif //GPURESOURCECACHE_H
