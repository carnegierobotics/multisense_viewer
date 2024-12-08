//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACERKERNELS_H
#define MULTISENSE_VIEWER_RAYTRACERKERNELS_H

#include <sycl/sycl.hpp>

namespace VkRender::RT::Kernels {

    class RenderKernel {
    public:
        RenderKernel(uint8_t* imageMemory, uint32_t width, uint32_t size)
            : m_imageMemory(imageMemory), m_width(width), m_size(size) {}

        void operator()(sycl::nd_item<2> item) const {
            uint32_t x = item.get_global_id(1); // Column index
            uint32_t y = item.get_global_id(0); // Row index

            // Compute pixel index (row-major order)
            uint32_t pixelIndex = (y * m_width + x) * 4; // RGBA8: 4 bytes per pixel

            // Ensure we don't go out of bounds
            if (pixelIndex < m_size) {
                m_imageMemory[pixelIndex + 0] = static_cast<uint8_t>(x % 256); // R
                m_imageMemory[pixelIndex + 1] = static_cast<uint8_t>(y % 256); // G
                m_imageMemory[pixelIndex + 2] = 0;                             // B
                m_imageMemory[pixelIndex + 3] = 255;                           // A
            }
        }

    private:
        uint8_t* m_imageMemory;
        uint32_t m_width;
        uint32_t m_size;
    };

}


#endif //MULTISENSE_VIEWER_RAYTRACERKERNELS_H
