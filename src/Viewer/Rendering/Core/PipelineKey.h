//
// Created by mgjer on 09/10/2024.
//

#ifndef PIPELINEKEY_H
#define PIPELINEKEY_H

#include <string>
#include <vector>
#include <filesystem>
#include <array>
#include <Viewer/Rendering/Components/MaterialComponent.h>
#include <vulkan/vulkan.h> // For Vulkan types

namespace VkRender {


    struct PipelineInfo {
        /* ───────── descriptor-set layouts (0..2) ─────── */
        std::vector<VkDescriptorSetLayout> setLayouts  = {};

        /* ───────── vertex format (binding 0+1) ───────── */
        std::vector<VkVertexInputBindingDescription> bindings{};
        std::vector<VkVertexInputAttributeDescription> attrs{};
        uint32_t attrCount = 0;

        MaterialInstance* materialInstance = nullptr;
    };
    struct PipelineKey
    {
        /* ───────── fixed-function state ───────── */
        AlphaMode           renderMode      = AlphaMode::Opaque;
        VkPrimitiveTopology  topology        = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPolygonMode        polygonMode     = VK_POLYGON_MODE_FILL;

        /* ───────── shader / material signature ───────── */
        uint32_t             meshId           = 0;        // 32-bit crc or hash of vertex shader path
        uint32_t             vsCRC           = 0;         // 32-bit crc or hash of vertex shader path
        uint32_t             fsCRC           = 0;         // 32-bit crc of fragment shader path
        uint32_t             materialFlags   = 0;         // e.g. bit0 = hasTexture, bit1 = alphaTest …

        bool operator==(const PipelineKey& other) const;

    };

    struct PipelineKeyHash
    {
        size_t operator()(const PipelineKey& k) const noexcept
        {
            size_t h = 0;
            auto mix = [&h](auto v)
            {
                h ^= std::hash<decltype(v)>{}(v) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
            };

            mix(k.renderMode);
            mix(k.topology);
            mix(k.polygonMode);
            mix(k.vsCRC);
            mix(k.fsCRC);
            mix(k.meshId);
            mix(k.materialFlags);

            return h;
        }
    };

};

#endif //PIPELINEKEY_H
