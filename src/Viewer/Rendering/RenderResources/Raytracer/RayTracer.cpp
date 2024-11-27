//
// Created by magnus on 11/27/24.
//

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracer.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracerKernels.h"
#include <sycl/sycl.hpp>

namespace VkRender::RT {

    void RayTracer::setup(std::shared_ptr<Scene> &scene) {
        m_scene = scene;

    }

    void RayTracer::update(uint32_t width, uint32_t height) {
        m_width = width;
        m_height = height;

        // Allocate image memory
        if (m_imageMemory) {
            delete[] static_cast<char*>(m_imageMemory);
        }
        m_imageMemory = new char[width * height * 4]; // Assuming RGBA8 image

        // Set up SYCL queue
        sycl::property_list properties{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling()};
        sycl::queue queue = sycl::queue(sycl::gpu_selector());
        Log::Logger::getInstance()->info("Using device: {}",queue.get_device().get_info<sycl::info::device::name>());

        for (const auto &device : sycl::device::get_devices()) {
            std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        }

        // Create buffers
        sycl::buffer<char, 1> imageBuffer(static_cast<char*>(m_imageMemory), sycl::range<1>(width * height * 4));

        // Collect meshes from the scene
        std::vector<Vertex> allVertices;
        std::vector<uint32_t> allIndices;
        std::vector<Kernels::MeshInfo> meshInfos; // Stores index offsets and counts for each mesh



        auto view = m_scene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto e : view) {
            Entity entity(e, m_scene.get());
            // Get mesh and transform components
            auto& meshComponent = entity.getComponent<MeshComponent>();
            auto& transformComponent = entity.getComponent<TransformComponent>();
            // Load mesh data
            MeshData meshData(meshComponent.m_type, meshComponent.m_meshPath);
            // Store the starting index and count for this mesh
            Kernels::MeshInfo meshInfo{};
            meshInfo.indexOffset = static_cast<uint32_t>(allIndices.size());
            meshInfo.indexCount = static_cast<uint32_t>(meshData.indices.size());
            meshInfo.transform = transformComponent.getTransform(); // Assuming it's a 4x4 matrix
            meshInfos.push_back(meshInfo);

            // Append vertices and indices
            size_t vertexOffset = allVertices.size();
            allVertices.insert(allVertices.end(), meshData.vertices.begin(), meshData.vertices.end());

            // Adjust indices for the vertex offset
            for (auto index : meshData.indices) {
                allIndices.push_back(static_cast<uint32_t>(index + vertexOffset));
            }
        }


        // Create buffers for vertices, indices, and mesh info
        sycl::buffer<Vertex, 1> vertexBuffer(allVertices.data(), sycl::range<1>(allVertices.size()));
        sycl::buffer<uint32_t, 1> indexBuffer(allIndices.data(), sycl::range<1>(allIndices.size()));
        sycl::buffer<Kernels::MeshInfo, 1> meshInfoBuffer(meshInfos.data(), sycl::range<1>(meshInfos.size()));

        // Submit kernel
        queue.submit([&](sycl::handler& cgh) {
            auto imageAcc = imageBuffer.get_access<sycl::access::mode::write>(cgh);
            auto vertexAcc = vertexBuffer.get_access<sycl::access::mode::read>(cgh);
            auto indexAcc = indexBuffer.get_access<sycl::access::mode::read>(cgh);
            auto meshInfoAcc = meshInfoBuffer.get_access<sycl::access::mode::read>(cgh);

            VkRender::RT::Kernels::rayTracingKernel(cgh, imageAcc, vertexAcc, indexAcc, meshInfoAcc, meshInfos.size(), width, height);
        });

        // Wait for the queue to finish
        queue.wait();
    }
}