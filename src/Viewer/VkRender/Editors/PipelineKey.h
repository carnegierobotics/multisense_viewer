//
// Created by mgjer on 09/10/2024.
//

#ifndef PIPELINEKEY_H
#define PIPELINEKEY_H

#include <string>
#include <vector>
#include <filesystem>
#include <vulkan/vulkan.h> // For Vulkan types

namespace VkRender {
    enum class RenderMode {
        Opaque,
        Transparent,
        Wireframe,
        // Add other render modes as needed
    };

    struct PipelineKey {
        RenderMode renderMode = RenderMode::Opaque;
        std::filesystem::path vertexShaderName = "default.vert";
        std::filesystem::path fragmentShaderName = "default.frag";
        std::vector<VkDescriptorSetLayout> setLayouts; // Include the descriptor set layout
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
        VkPolygonMode polygonMode = VK_POLYGON_MODE_MAX_ENUM;
        uint64_t* materialPtr = nullptr;
        bool operator==(const PipelineKey& other) const;
    };


}

template<>
struct std::hash<VkRender::PipelineKey> {
    std::size_t operator()(const VkRender::PipelineKey &key) const {
        std::size_t seed = 0;

        // Helper function to hash and combine with seed
        auto hash_combine = [&seed](auto&& value) {
            std::hash<std::decay_t<decltype(value)>> hasher;
            seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };

        // Hash each member of PipelineKey
        hash_combine(static_cast<int>(key.renderMode)); // Assuming RenderMode is an enum
        hash_combine(key.vertexShaderName);
        hash_combine(key.fragmentShaderName);
        hash_combine(key.materialPtr);
        hash_combine(key.setLayouts.size());
        hash_combine(static_cast<int>(key.topology)); // Assuming topology is an enum
        hash_combine(static_cast<int>(key.polygonMode)); // Assuming polygonMode is an enum

        return seed;
    }
};

#endif //PIPELINEKEY_H
