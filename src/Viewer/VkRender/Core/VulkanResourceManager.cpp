//
// Created by magnus on 7/29/24.
//


#include "Viewer/VkRender/Core/VulkanResourceManager.h"
namespace VkRender{
    VulkanResourceManager* VulkanResourceManager::instance = nullptr;
    std::once_flag VulkanResourceManager::initInstanceFlag;

}
