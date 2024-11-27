//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACERKERNELS_H
#define MULTISENSE_VIEWER_RAYTRACERKERNELS_H

#include <glm/glm.hpp>
#include <sycl/sycl.hpp>

#include "Viewer/Rendering/Core/RenderDefinitions.h"

namespace VkRender::RT::Kernels {

    struct MeshInfo {
        uint32_t indexOffset;
        uint32_t indexCount;
        // Assuming transform is a 4x4 matrix; define accordingly
        glm::mat4 transform;
    };

    void rayTracingKernel(
            sycl::handler& cgh,
            const sycl::accessor<char, 1, sycl::access::mode::write>& imageAcc,
            const sycl::accessor<Vertex, 1, sycl::access::mode::read>& vertexAcc,
            const sycl::accessor<uint32_t, 1, sycl::access::mode::read>& indexAcc,
            const sycl::accessor<MeshInfo, 1, sycl::access::mode::read>& meshInfoAcc,
            size_t numMeshes,
            uint32_t width,
            uint32_t height);
}


#endif //MULTISENSE_VIEWER_RAYTRACERKERNELS_H
