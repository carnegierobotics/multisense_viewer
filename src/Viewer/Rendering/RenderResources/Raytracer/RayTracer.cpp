//
// Created by magnus on 11/27/24.
//

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracer.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracerKernels.h"
#include <sycl/sycl.hpp>

#include "Viewer/Tools/SyclDeviceSelector.h"

namespace VkRender::RT {
    RayTracer::RayTracer(Application* ctx, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) : m_context(
        ctx) {
        m_scene = scene;
        m_width = width;
        m_height = height;
        m_camera = Camera(width, height);
        auto& queue = m_selector.getQueue();

        // Load the scene into gpu memory
        // Create image memory
        m_imageMemory = new uint8_t[width * height * 4]; // Assuming RGBA8 image

        m_gpu.imageMemory = sycl::malloc_device<uint8_t>(m_width * m_height * 4, queue);
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }

        std::vector<InputAssembly> vertexData;
        std::vector<uint32_t> indices;

        auto view = scene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto e : view) {
            Entity entity(e, scene.get());
            std::string tag = entity.getName();
            if (tag != "RayTracedObject")
                continue;
            auto& meshComponent = entity.getComponent<MeshComponent>();
            std::shared_ptr<MeshData> meshData = m_meshManager.getMeshData(meshComponent);
            for (auto& vert : meshData->vertices) {
                InputAssembly input{};
                input.position = vert.pos;
                input.color = vert.color;
                input.normal = vert.normal;
                vertexData.emplace_back(input);
            }
            indices = meshData->indices;
            m_gpu.numVertices = vertexData.size();
            m_gpu.numIndices = indices.size();
        }

        m_gpu.vertices = sycl::malloc_device<InputAssembly>(vertexData.size(), queue);
        queue.memcpy(m_gpu.vertices, vertexData.data(), vertexData.size() * sizeof(InputAssembly));

        m_gpu.indices = sycl::malloc_device<uint32_t>(indices.size(), queue);
        queue.memcpy(m_gpu.indices, indices.data(), indices.size() * sizeof(uint32_t));
        queue.wait();


        {
            auto view = m_scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
            for (auto e : view) {
                Entity entity(e, m_scene.get());
                auto& transform = entity.getComponent<TransformComponent>();
                // Retrieve focal point from the camera gizmo parameters
                auto cameraGizmoParams = std::dynamic_pointer_cast<CameraGizmoMeshParameters>(
                    entity.getComponent<MeshComponent>().meshParameters
                );
                float focalPoint = cameraGizmoParams->focalPoint;
                // Define your field of view here (in radians)
                // For example:
                float horizontalFOV = glm::radians(60.0f);
                float verticalFOV = glm::radians(40.0f);
                // The base forward vector pointing down the -Z axis
                glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
                // A helper lambda to create a ray entity
                auto createRay = [&](const std::string& name, float horizontalAngle, float verticalAngle) {
                    auto cornerEntity = m_scene->createEntity(name);
                    auto& mesh = cornerEntity.addComponent<MeshComponent>(CYLINDER);
                    cornerEntity.getComponent<TransformComponent>() = transform;
                    auto cylinderParams = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh.meshParameters);
                    cylinderParams->magnitude = 1.0f;
                    cylinderParams->origin = glm::vec3(0.0f, 0.0f, focalPoint);
                    // Rotate the forward vector by verticalAngle around X and horizontalAngle around Y
                    glm::mat4 pitch = glm::rotate(glm::mat4(1.0f), verticalAngle, glm::vec3(1.0f, 0.0f, 0.0f));
                    glm::mat4 yaw = glm::rotate(glm::mat4(1.0f), horizontalAngle, glm::vec3(0.0f, 1.0f, 0.0f));

                    glm::vec3 dir = glm::vec3(yaw * pitch * glm::vec4(forward, 0.0f));
                    cylinderParams->direction = glm::normalize(dir);
                };
                // Create rays for each corner:
                // Top-left corner: rotate upward (positive vertical angle) and to the left (negative horizontal angle)
                createRay("Top Left", -horizontalFOV * 0.5f, verticalFOV * 0.5f);
                // Top-right corner: upward and to the right (positive horizontal angle)
                createRay("Top Right", horizontalFOV * 0.5f, verticalFOV * 0.5f);
                // Bottom-left corner: downward (negative vertical angle) and left (negative horizontal angle)
                createRay("Bottom Left", -horizontalFOV * 0.5f, -verticalFOV * 0.5f);
                // Bottom-right corner: downward and right
                createRay("Bottom Right", horizontalFOV * 0.5f, -verticalFOV * 0.5f);
            }
        }
    }

    void RayTracer::update() {
        auto& queue = m_selector.getQueue();

        glm::vec3 cameraOrigin, cameraDirection;

        auto view = m_scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
        for (auto e : view) {
            Entity entity(e, m_scene.get());
            auto& camera = entity.getComponent<CameraComponent>();
            auto& transform = entity.getComponent<TransformComponent>();

            cameraOrigin = transform.getPosition();

            glm::quat rotation = transform.getRotationQuaternion();
            // Calculate the camera direction by rotating the default forward vector
            glm::vec3 defaultForward(0.0f, 0.0f, 1.0f); // Forward direction in OpenGL convention
            cameraDirection = glm::normalize(rotation * defaultForward);

            auto cameraGizmoParams = std::dynamic_pointer_cast<CameraGizmoMeshParameters>(
                entity.getComponent<MeshComponent>().meshParameters);

            float focalPoint = cameraGizmoParams->focalPoint;
            // TODO draw the camera generated vectors in mesh view.
        }

        uint32_t tileWidth = 16;
        uint32_t tileHeight = 16;
        sycl::range localWorkSize(tileHeight, tileWidth);
        sycl::range globalWorkSize(m_height, m_width);

        queue.submit([&](sycl::handler& h) {
            // Create a kernel instance with the required parameters
            const Kernels::RenderKernel kernel(m_gpu, m_width, m_height, m_width * m_height * 4, cameraOrigin,
                                               cameraDirection);
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
