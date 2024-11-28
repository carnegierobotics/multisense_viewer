//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RENDERCOMMAND_H
#define MULTISENSE_VIEWER_RENDERCOMMAND_H
#include <vulkan/vulkan_core.h>

#include "Viewer/Rendering/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/Rendering/MeshInstance.h"
#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Scenes/Entity.h"


namespace VkRender{
    enum class DescriptorManagerType : uint32_t;

    struct RenderCommand {
        Entity entity{};
        std::shared_ptr<DefaultGraphicsPipeline> pipeline = nullptr;
        MeshInstance* meshInstance = nullptr;             // GPU-specific mesh data
        MaterialInstance* materialInstance = nullptr;  // GPU-specific material data
        std::unordered_map<DescriptorManagerType, VkDescriptorSet> descriptorSets{}; // Add the descriptor set here
        std::unordered_map<DescriptorManagerType, std::vector<VkWriteDescriptorSet>> descriptorWrites{}; // Added for tracking resources management

    };

}

#endif //MULTISENSE_VIEWER_RENDERCOMMAND_H
