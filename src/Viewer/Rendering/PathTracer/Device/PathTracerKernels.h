//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERKERNELS_H
#define PATHTRACERKERNELS_H

#include <sycl/sycl.hpp>

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
            sycl::atomic_ref<unsigned int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    photonCount(d_sceneDesc->photonCount);

            photonCount.fetch_add(static_cast<unsigned int>(1));
        }


        SYCL_EXTERNAL static bool intersectBLAS(const Ray &rayO, uint32_t geomIdx, Hit &out, const SceneDesc &scene);

        SYCL_EXTERNAL static bool intersectScene(const Ray &rayW, Hit *hit, const SceneDesc &scene);

    private:
        SceneDesc *d_sceneDesc;
        SceneSettings d_sceneSettings;
        FrameBuffer d_framebuffer;

        SYCL_EXTERNAL void traceOnePhoton(uint32_t photonID) const;


        SYCL_EXTERNAL void castContributions(const float3 &hitPoint, const float &throughput) const;
    };
}
#endif //PATHTRACERKERNELS_H
