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
            RenderSettings settings = RenderSettings()
        ) : d_sceneDesc(scene), d_framebuffer(framebuffer), d_sceneSettings(settings) {
        }


        void operator()(sycl::item<1> item) const {
            uint32_t totalPhotonCount = item.get_range().get(0);
            // Each thread traces one photon.
            sycl::atomic_ref<unsigned int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    photonCount(d_sceneDesc->photonCount);

            uint64_t ID = photonCount.fetch_add(static_cast<unsigned int>(1));

            traceOnePhoton(ID, totalPhotonCount);
        }


        SYCL_EXTERNAL static bool intersectBLAS(const Ray &rayO, uint32_t geomIdx, Hit &out, const SceneDesc &scene);

        SYCL_EXTERNAL static bool intersectScene(const Ray &rayW, Hit *hit, const SceneDesc &scene);

    private:
        SceneDesc *d_sceneDesc;
        RenderSettings d_sceneSettings;
        FrameBuffer d_framebuffer;

        SYCL_EXTERNAL void traceOnePhoton(uint64_t photonID, uint32_t totalPhotonCount) const;


        SYCL_EXTERNAL void castContributions(
            const Hit& hitPoint,
            float throughput,
            const float3& surfaceNormal) const;
    };
}
#endif //PATHTRACERKERNELS_H
