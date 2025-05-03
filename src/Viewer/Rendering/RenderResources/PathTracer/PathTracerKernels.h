//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERKERNELS_H
#define PATHTRACERKERNELS_H

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
#endif

#include <sycl/sycl.hpp>


#include "Viewer/Rendering/RenderResources/PathTracer/GPUDataTypes.h"

namespace VkRender::PathTracer {
    // -------------------------
    // Kernel launcher class
    // -------------------------
    class PathTracerKernels {
    public:
        PathTracerKernels(
            sycl::queue &queue,
            const SceneGPUMesh &scene,
            uint32_t samplesPerPixel = 1
        ) : m_queue(queue), m_scene(scene), m_samplesPerPixel(samplesPerPixel) {

        }

        // Launches the SYCL kernel
        void renderFrame() {

            size_t total_tasks = 1000;

            m_queue.submit([&](sycl::handler &cgh) {
                auto out = sycl::stream(1024, 768, cgh);
                auto scene = m_scene;

                Camera cam;

                cgh.parallel_for(sycl::range<1>(total_tasks), [cam, scene, out](sycl::item<1> it) {
                    size_t task_id = it.get_id(0);



                    out << "intel_get_cycle_counter: ";

            #ifdef __SYCL_DEVICE_ONLY__
                    ulong cycle_counter = intel_get_cycle_counter();
                    out << cycle_counter << endl;
            #endif
                });
            }).wait();

            std::cout << "Done" << std::endl;

        }

    private:
        sycl::queue m_queue;
        SceneGPUMesh    m_scene;
        uint32_t    m_samplesPerPixel;

        bool intersectAABB(const float minB[3], const float maxB[3]) const;
        bool intersectTri(const Vertex &v0, const Vertex &v1, const Vertex &v2, float &t) const;
    };


}

#endif //PATHTRACERKERNELS_H
