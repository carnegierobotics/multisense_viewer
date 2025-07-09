//
// Created by magnus on 5/2/25.
//

#ifndef PathTracerSetup_H
#define PathTracerSetup_H

#include <Viewer/Rendering/Editors/ArcballCamera.h>
#include <Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h>
#include <Viewer/Tools/SYCLDeviceSelector.h>

#include "Viewer/Rendering/Core/VulkanTexture.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/PathTracer/Device/PathTracerKernels.h"

// -------------------------
// Main tracer class
// -------------------------
namespace VkRender::PathTracer {
    /// World‑space cube vertices + indices for one BVH node
    enum class BVHLevel : uint32_t // 0 = BLAS, 1 = TLAS  (room for more levels)
    {
        BLAS = 0,
        TLAS = 1
    };

    struct alignas(16) DebugBound {
        glm::mat4 model; // world transform (translation only)
        glm::vec3 size; // width / height / depth
        uint32_t level; // cast from BVHLevel – lets you filter later
        std::string name; // human-readable mesh name (or instance identifier)
    };

    struct PathTracerSetupCreateInfo {
        sycl::queue &queue;
        std::shared_ptr<SYCLDeviceSelector> device;
        uint32_t framebufferSize = 960 * 600 * 10; // 10 images of size 960x600x4

        PathTracerSetupCreateInfo() = delete;

        explicit
        PathTracerSetupCreateInfo(std::shared_ptr<SYCLDeviceSelector> dev) : device(dev), queue(dev->getQueue()) {
        }
    };

    struct EditorCamera {
        const ArcballCamera *camera = nullptr;
        uint32_t editorWidth = 1024;
        uint32_t editorHeight = 768;
        bool movedSinceLastFrame = false;
    };

    struct RenderInfoOutput {
        uint32_t frameID = 0;
        uint64_t photonCount = 0;
    };
    class PathTracerSetup {
    public:
        explicit PathTracerSetup(const PathTracerSetupCreateInfo &createInfo) : m_queue(createInfo.queue),
                                                                              m_createInfo(createInfo) {
            std::memset(&m_sceneDescHost, 0, sizeof(m_sceneDescHost));
            std::memset(&m_sceneDescDevice, 0, sizeof(m_sceneDescDevice));

            // setup output'
            setupFrameBuffers();
        }

        ~PathTracerSetup();

        /** (re)allocates all GPU buffers that depend on scene topology */
        void uploadScene(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera = EditorCamera());

        void traverseBVH();

        void intersectBVH(uint32_t nodeIndex);

        /** reallocates output image if #cameras / resolution changed */
        void setupFrameBuffers();

        /** per‑frame fast update of transforms, animated emissive, … */
        void updateDynamic(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera);

        void resetImageMemory();

        /** launches photon + contribution kernels */
        RenderInfoOutput renderFrame(const RenderSettings &settings);

        RenderInfoOutput radiativeBackprop(std::shared_ptr<EditorPathTracerLayerUI> imageUI, const RenderSettings &settings);

        /** copies the device framebuffer back to host */
        void generateImages(std::shared_ptr<EditorPathTracerLayerUI> ptr);

        void writePhotonPLY(const std::vector<Photon> &photons, const std::string& filename = "photons.ply");

        void generateEditorImage(const std::shared_ptr<VulkanTexture2D> &viewportTexture, float gamma, float exposure);

        void createEditorCamera(const std::shared_ptr<ArcballCamera> &camera, int32_t int32, int32_t height);

        const PathTracerSetupCreateInfo &getCreateInfo() { return m_createInfo; }

        std::vector<BVHNode> getBLASNodes() { return m_blasNodes; }
        std::vector<TLASNode> getTLASNodes() { return m_tlasNodes; }
        SceneDesc& getSceneDescription() { return m_sceneDescHost; }
        void printMaterialIDs();

    private:
        /*--- helpers called only from uploadScene() ---*/
        void collectGeometry(const std::shared_ptr<Scene> &scene);

        void collectInstances(const std::shared_ptr<Scene> &scene);

        void collectLights(const std::shared_ptr<Scene> &scene);

        void collectCameras(const std::shared_ptr<Scene> &scene, EditorCamera editorCamera);


        void buildSceneDesc();

        void buildBLASForAllMeshes();
        void buildBLASForPointCloud();
        void buildTopLevelBVH();

        /*--- device clean‑up ---*/
        void freeDeviceMemory();

        // Helper to Allocate less verbose
        // size_t num bytes of type T
        template<typename T>
        T *deviceAlloc(size_t num) {
            return static_cast<T *>(sycl::malloc_device(sizeof(T) * num, m_queue));
        }
        /* ---------------Runtime settings------------------*/
        uint32_t m_frameID = 0;
        uint64_t m_totalPhotons = 0;
        uint32_t m_photonCount = 0;

        /*----------------------------------------------*/
        sycl::queue m_queue;
        PathTracerSetupCreateInfo m_createInfo;

        FrameBuffer m_frameBuffers{};
        FrameBuffer d_frameBuffers{};
        // DEBUG NAMES
        std::vector<std::string> m_meshNames; ///< same length as m_meshRanges
        std::vector<std::string> m_blasNames; ///< same length as m_blasRanges
        std::vector<std::string> m_materialNames;

        // BVH
        //std::vector<BVHNode> m_bvhNodes;


        // === Member (host) staging arrays ===
        std::vector<Vertex> m_vertices;
        std::vector<Triangle> m_tris;
        std::vector<MeshRange> m_meshRanges;
        std::vector<Instance> m_instances;
        std::vector<Transform> m_transforms;
        std::vector<Material> m_materials;
        std::vector<MeshLight> m_lights;
        std::vector<Camera> m_cameras;

        std::vector<OrientedPoint> m_points;
        std::vector<PointCloudRange> m_pointRanges;
        // === Host staging helper variables ===
        std::unordered_map<std::string, uint32_t> m_meshIndexMap;

        // CPU copies
        std::vector<BVHNode> m_betaNodes;
        std::vector<BLASRange> m_betaRanges;
        std::vector<BVHNode> m_blasNodes;
        std::vector<BLASRange> m_blasRanges;
        std::vector<TLASNode> m_tlasNodes;
        std::vector<uint32_t> m_trianglePerm; // BVH triangle permutation
        std::vector<Photon> m_photonHits; // Map of photons hits
        uint32_t m_photonHitCounter; // Map of photons hits
        // device pointers
        BVHNode *d_blasNodes = nullptr;
        TLASNode *d_tlasNodes = nullptr;
        BLASRange *d_blasRanges = nullptr;
        BVHNode * d_betaNodes = nullptr;
        BLASRange* d_betaRanges = nullptr;
        uint32_t* d_trianglePerm = nullptr;
        Photon* d_photonHits = nullptr;
        uint32_t* d_photonHitCounter = nullptr;

        // === Device USM pointers ===
        OrientedPoint *d_points = nullptr;
        Vertex *d_vertices = nullptr;
        Triangle *d_tris = nullptr;
        MeshRange *d_meshRanges = nullptr;
        Instance *d_instances = nullptr;
        Transform *d_transforms = nullptr;
        Material *d_materials = nullptr;
        MeshLight *d_lights = nullptr;
        Camera *d_cameras = nullptr;
        SceneDesc *d_sceneDesc = nullptr;
        // Host copy of the descriptor used to build the device-side struct
        SceneDesc  m_sceneDescDevice;   // device pointers (was m_sceneDesc)
        SceneDesc  m_sceneDescHost;     // host  pointers – new

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

        // === Radiative Backpropagation ===
        std::vector<ParamOffset> m_kdOffset;   // size = m_materials.size()
        std::vector<ParamOffset> m_vtxOffset;  // size = m_vertices.size()
        ParamOffset * d_kdOffset;
        ParamOffset * d_vtxOffset;
        float      * d_gradAll;   // ← the one big vector
        float      * d_gradKdImage;   // deub gimage
        uint32_t                  m_totalParamSlots = 0;
        uint32_t m_backpropIterations = 0;

    };
} // VkRender

#endif //PathTracerSetup_H
