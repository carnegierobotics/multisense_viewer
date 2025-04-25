//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHINSTANCE_H
#define MULTISENSE_VIEWER_MESHINSTANCE_H

#include "Viewer/Rendering/MeshData.h"
#include "Viewer/Rendering/Core/PipelineKey.h"

namespace VkRender {
    enum class DescriptorManagerType : uint32_t;
    struct MaterialInstance;

    struct MeshInstance {
        std::unique_ptr<Buffer> vertexBuffer{};
        std::unique_ptr<Buffer> indexBuffer{};
        std::unique_ptr<Buffer> instanceBuffer{};               // VkBuffer w/ VK_VERTEX_INPUT_RATE_INSTANCE

        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t instanceCount = 0;
        uint32_t drawCount = 0;
        bool SSBO = false;
        // Additional data like vertex layout, primitive type, etc.
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        MeshDataType m_type{};
        bool usesVertexBuffers = false;
        uint32_t lastUpdatedVersion = 0;


        void ensureInstanceBuffer(VkDeviceSize size, VulkanDevice& dev);

    };

    struct InstanceData {           // 64 B, aligned to std140
        glm::mat4 model;            // you can add color, id, etc. later
    };

    struct InstanceBatch {
        std::shared_ptr<MeshInstance>      mesh;
        std::shared_ptr<MaterialInstance>  material;
        std::unordered_map<DescriptorManagerType, VkDescriptorSet> sets;
        std::vector<InstanceData>          cpuInstances;   // per-frame
        PipelineKey                        key;
    };

}

#endif //MULTISENSE_VIEWER_MESHINSTANCE_H
