//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERKERNELS_H
#define PATHTRACERKERNELS_H

#include <sycl/sycl.hpp>

#include "KernelHelpers.h"
#include "Viewer/Rendering/PathTracer/GPUDataTypes.h"


#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

namespace VkRender::PathTracer {
    class PathTracerMeshKernel {
    public:
        PathTracerMeshKernel(
            SceneDesc *scene,
            FrameBuffer framebuffer,
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
        FrameBuffer d_framebuffer;

        SYCL_EXTERNAL void traceOnePhoton(uint32_t photonID) const;

        SYCL_EXTERNAL bool intersectBLAS(const Ray &rayO, uint32_t geomIdx, Hit &out) const;

        SYCL_EXTERNAL bool intersectScene(const Ray &rayW, Hit *hit) const;

        SYCL_EXTERNAL void castContributions(const float3 &hitPoint, const float &throughput) const;
    };
}
#endif //PATHTRACERKERNELS_H
