//
// Created by mgjer on 09/10/2024.
//

#ifndef PIPELINEKEY_H
#define PIPELINEKEY_H

#include <string>
#include <vulkan/vulkan.h> // For Vulkan types

namespace VkRender {
    enum class RenderMode {
        Opaque,
        Transparent,
        Wireframe,
        // Add other render modes as needed
    };

    struct PipelineKey {
        RenderMode renderMode;
        std::string shaderName;
        VkDescriptorSetLayout descriptorSetLayout; // Include the descriptor set layout

        bool operator==(const PipelineKey& other) const;
    };


}

template<>
struct std::hash<VkRender::PipelineKey> {
    std::size_t operator()(const VkRender::PipelineKey &key) const {
        // Compute individual hashes for each member and combine them
        std::size_t hash1 = std::hash<int>{}(static_cast<int>(key.renderMode)); // Assuming RenderMode is an enum
        std::size_t hash2 = std::hash<std::string>{}(key.shaderName);
        std::size_t hash3 = std::hash<VkDescriptorSetLayout>{}(key.descriptorSetLayout);

        // Combine the hash values (bitwise XOR and shifts)
        return hash1 ^ (hash2 << 1) ^ (hash3 << 2);
    }
};

#endif //PIPELINEKEY_H
