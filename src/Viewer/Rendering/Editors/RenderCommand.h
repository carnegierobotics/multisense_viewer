//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RENDERCOMMAND_H
#define MULTISENSE_VIEWER_RENDERCOMMAND_H

namespace VkRender{
    struct RenderCommand {
        Entity entity{};
        std::shared_ptr<DefaultGraphicsPipeline> pipeline = nullptr;
        MeshInstance* meshInstance = nullptr;             // GPU-specific mesh data
        MaterialInstance* materialInstance = nullptr;  // GPU-specific material data
        PointCloudInstance* pointCloudInstance = nullptr;  // GPU-specific material data
        std::unordered_map<uint32_t, VkDescriptorSet> descriptorSets{}; // Add the descriptor set here

    };

}

#endif //MULTISENSE_VIEWER_RENDERCOMMAND_H
