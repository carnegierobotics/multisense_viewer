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

#include "Viewer/Rendering/PathTracer/GPUDataTypes.h"
#include "Viewer/Rendering/PathTracer/PathTracerTypes.h"

namespace VkRender::PathTracer {
    class PathTracerMeshKernel {
    public:
        PathTracerMeshKernel(
            SceneDesc *scene,
            FrameBuffer* framebuffer,
            SceneSettings settings = SceneSettings()
        ) : d_sceneDesc(scene), d_framebuffer(framebuffer), d_sceneSettings(settings) {
        }


        void operator()(sycl::item<1> item) const {
            size_t photonID = item.get_linear_id();

            // Each thread traces one photon.
            traceOnePhoton(photonID + d_sceneDesc->photonCount);
            d_sceneDesc->photonCount++;
        }

    private:
        SceneDesc *d_sceneDesc;
        SceneSettings d_sceneSettings;
        FrameBuffer *d_framebuffer;

        void traceOnePhoton(uint32_t photonID) const;

        bool intersectBLAS(const Ray &ray, uint32_t meshIdx, Hit &out) const;

        bool intersectScene(const Ray &rayW, Hit *hit) const;

        void castContributions(const float3 &hitPoint, const float &throughput) const;

        // your per-photon RNG: e.g. hashed by photonID
    };
}
#endif //PATHTRACERKERNELS_H
