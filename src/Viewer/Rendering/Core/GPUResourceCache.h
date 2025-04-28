//
// Created by mgjer on 27/04/2025.
//

#ifndef GPURESOURCECACHE_H
#define GPURESOURCECACHE_H


#include "Viewer/Rendering/Core/ShaderModuleCache.h"

namespace VkRender {

class GPUResourceCache {
public:
  explicit GPUResourceCache(VulkanDevice& device)
    : shaderModules(device.m_LogicalDevice)
  {}

  ShaderModuleCache      shaderModules;
  //PipelineCache          pipelines;
  //DescriptorSetCache     descriptors;
  //TextureCache textures;

  // ... other GPU‐side caches (framebuffers, samplers, etc.) ...
};

} // namespace VkRender

#endif //GPURESOURCECACHE_H
