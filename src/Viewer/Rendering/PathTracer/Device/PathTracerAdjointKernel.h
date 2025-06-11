//
// Created by magnus on 6/11/25.
//

#ifndef PATHTRACERADJOINTKERNEL_H
#define PATHTRACERADJOINTKERNEL_H


#include <sycl/sycl.hpp>

#include "KernelHelpers.h"

#include "Viewer/Rendering/PathTracer/GPUDataTypes.h"


#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif


namespace VkRender::PathTracer {
    class PathTracerAdjointKernel {
    public:
        PathTracerAdjointKernel(
            SceneDesc *scene,
            FrameBuffer framebuffer,
            RenderSettings settings = RenderSettings()
        ) : d_sceneDesc(scene), d_framebuffer(framebuffer), d_sceneSettings(settings) {
        }


            void operator()(sycl::nd_item<2> it) const {
                const int py = it.get_global_id(0);
                const int px = it.get_global_id(1);
                if (px >= d_sceneDesc->cameras[1].width ||
                    py >= d_sceneDesc->cameras[1].height)
                    return;

                traceAdjoint(px, py);
        }


        SYCL_EXTERNAL static bool intersectBLASMesh(const Ray &rayO, uint32_t geomIdx, Hit &out, const SceneDesc &scene);
        SYCL_EXTERNAL static bool intersectBLASQuadric(const Ray &rayO, uint32_t geomIdx, Hit &out, const SceneDesc &scene);

        SYCL_EXTERNAL static bool intersectScene(const Ray &rayW, Hit *hit, const SceneDesc &scene);

    private:
        SceneDesc *d_sceneDesc;
        RenderSettings d_sceneSettings;
        FrameBuffer d_framebuffer;

        void traceAdjoint(int px, int py) const;

        // atomicAdd to the big vector (handles scalar or vector parameters)
        template<int Dim>
        SYCL_EXTERNAL
        inline void addToGradBuffer(float* gradAll,
                                    uint32_t baseSlot,
                                    const float (&value)[Dim]) const
        {
            for (int i = 0; i < Dim; ++i) {
                sycl::atomic_ref<float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    cell(gradAll[baseSlot + i]);
                cell += value[i];
            }
        }

    };
}




#endif //PATHTRACERADJOINTKERNEL_H
