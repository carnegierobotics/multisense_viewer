//
// Created by magnus on 5/2/25.
//

#ifndef PATHTRACERSYCL_H
#define PATHTRACERSYCL_H

#include <Viewer/Rendering/Editors/ArcballCamera.h>

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/PathTracer/PathTracerKernels.h"

// -------------------------
// Main tracer class
// -------------------------
namespace VkRender {

    struct PathTracerSYCLCreateInfo {
        sycl::queue queue;
        uint32_t framebufferSize = 960 * 600 * 4 * 10; // 10 images of size 960x600x4
    };

    struct EditorCamera {
        const ArcballCamera* camera = nullptr;
        uint32_t editorWidth = 1024;
        uint32_t editorHeight = 768;
    };

    class PathTracerSYCL {
    public:
        explicit PathTracerSYCL(const PathTracerSYCLCreateInfo& createInfo) : m_queue(createInfo.queue), m_createInfo(createInfo) {
            std::memset(&m_sceneDesc, 0, sizeof(m_sceneDesc));

            // setup output'
            setupFrameBuffers();
        }

        ~PathTracerSYCL();

        /** (re)allocates all GPU buffers that depend on scene topology */
        void uploadScene(const std::shared_ptr<Scene>& scene, EditorCamera editorCamera = EditorCamera());

        /** reallocates output image if #cameras / resolution changed */
        void setupFrameBuffers();

        /** per‑frame fast update of transforms, animated emissive, … */
        void updateDynamic(const std::shared_ptr<Scene>& scene);

        /** launches photon + contribution kernels */
        void renderFrame();

        /** copies the device framebuffer back to host */
        void generateImages(std::span<std::byte> outRGBA32f);

        void createEditorCamera(const std::shared_ptr<ArcballCamera> & camera, int32_t int32, int32_t height);

    private:
        /*--- helpers called only from uploadScene() ---*/
        void collectGeometry(const std::shared_ptr<Scene>& scene);

        void collectInstances(const std::shared_ptr<Scene>& scene);

        void collectLights(const std::shared_ptr<Scene>& scene);

        void collectCameras(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera);

        void buildSceneDesc();

        static void buildBVHNodes(
            const std::vector<PathTracer::Triangle> &triangles,
            const std::vector<float> &px,
            const std::vector<float> &py,
            const std::vector<float> &pz,
            std::vector<PathTracer::BVHNode> &outNodes);

        /*--- device clean‑up ---*/
        void freeDeviceMemory();

        // Helper to Allocate less verbose
        // size_t num bytes of type T
        template<typename T>
        T *deviceAlloc(size_t num) {
            return static_cast<T *>(sycl::malloc_device(sizeof(T) * num, m_queue));
        }

        /*----------------------------------------------*/
        sycl::queue m_queue;
        PathTracerSYCLCreateInfo m_createInfo;

        PathTracer::FrameBuffer m_frameBuffers{};
        PathTracer::FrameBuffer d_frameBuffers{};

        // === Member (host) staging arrays ===
        std::vector<float> m_px, m_py, m_pz;
        std::vector<float> m_nx, m_ny, m_nz;
        std::vector<PathTracer::Triangle> m_tris;
        std::vector<PathTracer::MeshRange> m_meshRanges;
        std::vector<PathTracer::BVHNode> m_bvh;
        std::vector<PathTracer::Instance> m_instances;
        std::vector<PathTracer::Transform> m_transforms;
        std::vector<PathTracer::Material> m_materials;
        std::vector<PathTracer::MeshLight> m_lights;
        std::vector<PathTracer::Camera> m_cameras;

        // === Host staging helper variables ===
        std::unordered_map<std::string, uint32_t> m_meshIndexMap;


        // === Device USM pointers ===
        float *d_px = nullptr;
        float *d_py = nullptr;
        float *d_pz = nullptr;
        float *d_nx = nullptr;
        float *d_ny = nullptr;
        float *d_nz = nullptr;
        PathTracer::Triangle *d_tris = nullptr;
        PathTracer::MeshRange *d_meshRanges = nullptr;
        PathTracer::BVHNode *d_bvh = nullptr;
        PathTracer::Instance *d_instances = nullptr;
        PathTracer::Transform *d_transforms = nullptr;
        PathTracer::Material *d_materials = nullptr;
        PathTracer::MeshLight *d_lights = nullptr;
        PathTracer::Camera *d_cameras = nullptr;
        PathTracer::SceneDesc *d_sceneDesc = nullptr;

        // Host copy of the descriptor used to build the device-side struct
        PathTracer::SceneDesc m_sceneDesc;

        // === Counts (optional mirrors) ===
        uint32_t m_triCount = 0;
        uint32_t m_meshCount = 0;
        uint32_t m_pointCount = 0;
        uint32_t m_pointCloudCount = 0;
        uint32_t m_instanceCount = 0;
        uint32_t m_transformCount = 0;
        uint32_t m_materialCount = 0;
        uint32_t m_lightCount = 0;
        uint32_t m_cameraCount = 0;
    };
} // VkRender

#endif //PATHTRACERSYCL_H
