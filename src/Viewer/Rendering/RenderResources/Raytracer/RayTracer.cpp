//
// Created by magnus on 11/27/24.
//

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracer.h"

#include <queue>

#include "Viewer/Rendering/RenderResources/Raytracer/RayTracerKernels.h"
#include <sycl/sycl.hpp>

#include "Viewer/Tools/SyclDeviceSelector.h"

namespace VkRender::RT {
    RayTracer::RayTracer(std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) {
        m_scene = scene;
        m_width = width;
        m_height = height;
        auto& queue = m_selector.getQueue();

        // Load the scene into gpu memory
        // Create image memory
        m_imageMemory = new uint8_t[width * height * 4]; // Assuming RGBA8 image

        m_gpu.imageMemory = sycl::malloc_device<uint8_t>(m_width * m_height * 4, queue);
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }
    }

    void RayTracer::update() {
        auto& queue = m_selector.getQueue();

        uint32_t tileWidth = 16;
        uint32_t tileHeight = 16;
        sycl::range<2> localWorkSize(tileHeight, tileWidth);
        sycl::range<2> globalWorkSize(m_height, m_width);

        queue.submit([&](sycl::handler& h) {
            // Create a kernel instance with the required parameters
            const Kernels::RenderKernel kernel(m_gpu.imageMemory, m_width, m_width * m_height * 4);
            h.parallel_for<class RenderKernel>(
                sycl::nd_range<2>(globalWorkSize, localWorkSize), kernel);
        }).wait();

        queue.memcpy(m_imageMemory, m_gpu.imageMemory, m_width * m_height * 4);
        queue.wait();

        saveAsPPM("sycl.ppm");;
    }

    RayTracer::~RayTracer() {
        if (m_imageMemory) {
            delete[] m_imageMemory;
        }

        if (m_gpu.imageMemory) {
            sycl::free(m_gpu.imageMemory, m_selector.getQueue());
        }
    }

    void RayTracer::saveAsPPM(const std::filesystem::path& filename) const {
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename.string());
        }

        // Write the PPM header
        file << "P6\n" << m_width << " " << m_height << "\n255\n";

        // Write pixel data in RGB format
        for (uint32_t y = 0; y < m_height; ++y) {
            for (uint32_t x = 0; x < m_width; ++x) {
                uint32_t pixelIndex = (y * m_width + x) * 4; // RGBA8: 4 bytes per pixel

                // Extract R, G, B components (ignore A)
                file.put(m_imageMemory[pixelIndex + 0]); // R
                file.put(m_imageMemory[pixelIndex + 1]); // G
                file.put(m_imageMemory[pixelIndex + 2]); // B
            }
        }

        file.close();
    }
}
