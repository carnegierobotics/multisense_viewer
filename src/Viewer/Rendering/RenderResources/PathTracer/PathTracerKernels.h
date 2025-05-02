//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERKERNELS_H
#define PATHTRACERKERNELS_H

#include <sycl/sycl.hpp>


#include "Viewer/Rendering/RenderResources/PathTracer/GPUDataTypes.h"

namespace VkRender::PathTracer {

    // -------------------------
    // Kernel launcher class
    // -------------------------
    class PathTracerKernels {
    public:
        PathTracerKernels(sycl::queue &q,
                          Vertex* d_vertices, size_t vCount,
                          Triangle* d_tris,    size_t tCount,
                          Material* d_mats,
                          BVHNode*  d_bvh,      size_t bvhCount,
                          uint8_t*  d_image,
                          uint32_t  width, uint32_t height)
            : queue(q),
              vertices(d_vertices), tris(d_tris), materials(d_mats), bvh(d_bvh),
              img(d_image), W(width), H(height)
        {}

        void renderFrame() {
            sycl::range<2> gws{H, W};
            queue.submit([&](sycl::handler &cgh){
                auto v      = vertices;
                auto t      = tris;
                auto m      = materials;
                auto n      = bvh;
                auto imgPtr = img;
                uint32_t w  = W;
                uint32_t h  = H;
                cgh.parallel_for(gws, [=](sycl::item<2> it){
                    uint32_t y = it.get_id(0);
                    uint32_t x = it.get_id(1);
                    size_t idx = (size_t(y) * w + x) * 4;
                    // 1) generate camera ray
                    // 2) traverse BVH 'n', intersect triangles 't', fetch vertices 'v'
                    // 3) shade via material 'm'
                    uint8_t r = 0, g = 0, b = 0, a = 255;
                    imgPtr[idx + 0] = r;
                    imgPtr[idx + 1] = g;
                    imgPtr[idx + 2] = b;
                    imgPtr[idx + 3] = a;
                });
            }).wait();
        }

    private:
        sycl::queue &queue;
        Vertex*    vertices;
        Triangle*  tris;
        Material*  materials;
        BVHNode*   bvh;
        uint8_t*   img;
        uint32_t   W, H;
    };


}

#endif //PATHTRACERKERNELS_H
