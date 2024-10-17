//
// Created by mgjer on 12/10/2024.
//

#ifndef POINTCLOUDCOMPONENT_H
#define POINTCLOUDCOMPONENT_H

#include <string>

#include "Viewer/VkRender/Core/VulkanTexture.h"
namespace VkRender {
    struct PointCloudComponent {
        std::string name;
        float pointSize = 1.0f;

        // File loading info
        bool usesVideoSource = false;
        std::filesystem::path depthVideoFolderSource = "path/to/images";
        std::filesystem::path colorVideoFolderSource = "path/to/images";
    };

    struct PointCloudInstance {
        struct Textures {
            std::shared_ptr<VulkanTexture2D> depth;
            std::shared_ptr<VulkanTexture2D> color;
            std::shared_ptr<VulkanTexture2D> chromaV;
            std::shared_ptr<VulkanTexture2D> chromaU;
        };
        std::vector<Textures> textures;
    };
}
#endif //POINTCLOUDCOMPONENT_H
