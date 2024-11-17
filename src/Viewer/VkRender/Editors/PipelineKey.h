//
// Created by mgjer on 09/10/2024.
//

#ifndef PIPELINEKEY_H
#define PIPELINEKEY_H

#include <string>
#include <vector>
#include <filesystem>
#include <array>  // Required for std::array
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
        std::vector<VkDescriptorSetLayout> setLayouts = {}; // Include the descriptor set layout, fixed size as we need to map to pre-compiled shaders
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
        VkPolygonMode polygonMode = VK_POLYGON_MODE_MAX_ENUM;
        uint64_t* materialPtr = nullptr;

        std::vector<VkVertexInputBindingDescription> vertexInputBindingDescriptions = {}; // TODO include in hash
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes; // TODO include in hash
        bool useCustomVertexInputBindings = false;
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
        hash_combine(static_cast<int>(key.renderMode));
        hash_combine(key.vertexShaderName);
        hash_combine(key.fragmentShaderName);
        hash_combine(key.materialPtr);
        hash_combine(key.setLayouts.size());
        hash_combine(static_cast<int>(key.topology));
        hash_combine(static_cast<int>(key.polygonMode));

        // Hash each attribute in vertexInputBindingDescriptions
        for (const auto& attr : key.vertexInputBindingDescriptions) {
            hash_combine(attr.binding);
            hash_combine(attr.stride);
            hash_combine(attr.inputRate);
        }

        // Hash each attribute in vertexInputAttributes
        for (const auto& attr : key.vertexInputAttributes) {
            hash_combine(attr.binding);
            hash_combine(attr.location);
            hash_combine(attr.format);
            hash_combine(attr.offset);
        }

        return seed;
    }
};

#endif //PIPELINEKEY_H
