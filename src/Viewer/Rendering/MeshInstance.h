//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHINSTANCE_H
#define MULTISENSE_VIEWER_MESHINSTANCE_H

#include "Viewer/Rendering/MeshData.h"

namespace VkRender {
    struct MeshInstance {
        std::unique_ptr<Buffer> vertexBuffer{};
        std::unique_ptr<Buffer> indexBuffer{};
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t drawCount = 0;
        bool SSBO = false;
        // Additional data like vertex layout, primitive type, etc.
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        MeshDataType m_type{};
        bool usesVertexBuffers = false;

    };
}

#endif //MULTISENSE_VIEWER_MESHINSTANCE_H
