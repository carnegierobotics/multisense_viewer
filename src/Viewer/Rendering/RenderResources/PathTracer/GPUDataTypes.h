//
// Created by magnus on 5/2/25.
//

#ifndef GPUDATATYPES_H
#define GPUDATATYPES_H

#include <cstdint>

namespace VkRender::PathTracer {

    // -------------------------
    // Basic GPU data types
    // -------------------------
    struct Vertex {
        float x, y, z;
        float nx, ny, nz;
        float u, v;
    };

    struct Triangle {
        uint32_t i0, i1, i2;
        uint32_t materialID;
    };

    struct Material {
        float diffuse[3];
        float emission[3];
    };

    struct BVHNode {
        float bboxMin[3];
        float bboxMax[3];
        int left;
        int right;
    };



}

#endif //GPUDATATYPES_H
