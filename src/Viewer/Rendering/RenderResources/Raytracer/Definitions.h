//
// Created by magnus-desktop on 12/8/24.
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <glm/glm.hpp>

namespace VkRender::RT{
    struct InputAssembly {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec3 normal;
    };

    struct GPUData {
        uint8_t* imageMemory = nullptr;
        InputAssembly* vertices;
        uint32_t*  indices; // e.g., {0, 1, 2, 2, 3, 0, ...}
        size_t numVertices;
        size_t numIndices;
    };



}
#endif //DEFINITIONS_H
