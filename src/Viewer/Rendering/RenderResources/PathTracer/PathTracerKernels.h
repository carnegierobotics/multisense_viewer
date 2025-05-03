//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERKERNELS_H
#define PATHTRACERKERNELS_H

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
#endif

#include <sycl/sycl.hpp>
#include <cmath>

#include "Viewer/Rendering/RenderResources/PathTracer/GPUDataTypes.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracerTypes.h"

namespace VkRender::PathTracer {
    class PathTracerMeshKernel {
    public:
        PathTracerMeshKernel(
            SceneDesc *scene
        ) : d_sceneDesc(scene) {
        }


        void operator()(sycl::item<1> item) const {
            size_t photonID = item.get_linear_id();

            // Each thread traces one photon.
            traceOnePhoton(photonID);
        }

    private:
        SceneDesc *d_sceneDesc;
        SceneSettings d_sceneSettings;
        FrameBuffer *d_frameBuffer;

        void traceOnePhoton(uint32_t photonID) const;

        // intersect the ray against the BVH + triangles
        bool intersectBVH(const Ray &ray, Hit *hit) const;

        void castContributions(const float3 &hitPoint, const float3 &throughput) const;

        // your per-photon RNG: e.g. hashed by photonID
    };
}
#endif //PATHTRACERKERNELS_H
