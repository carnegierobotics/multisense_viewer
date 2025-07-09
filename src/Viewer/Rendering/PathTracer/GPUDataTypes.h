//
// Created by magnus on 5/2/25.
//

#ifndef GPUDATATYPES_H
#define GPUDATATYPES_H

#include <cstdint>

#include <Viewer/Scenes/Entity.h>
#include "PathTracerTypes.h"

namespace VkRender {
    struct QuadricCloudAsset;
}

namespace VkRender::PathTracer {


    /*────────────────────────────────────────────────────────────────────────────*/
    /*  Helper macro – verify every struct is 16‑byte aligned & sized             */
    /*────────────────────────────────────────────────────────────────────────────*/
#define CHECK_16(T) static_assert(alignof(T)==16 && sizeof(T)%16==0,           \
                                 "" #T " must be 16‑byte aligned & sized")

    /*************************  Core Geometry *************************/
    struct alignas(16) Vertex {
        float3 pos; // 16 B
        float3 norm; // 16 B
    };

    CHECK_16(Vertex);

    struct alignas(16) Triangle {
        uint32_t v0{}, v1{}, v2{}; // 12 B
        uint32_t _pad0{}; // 16 B     ← keeps centroid 16‑aligned
        float3 centroid; // 16 B
        uint32_t _pad1{}; // 32 B ↦ sizeof()==32
    };

    CHECK_16(Triangle);

    enum PointType : uint32_t {
        Gaussian2DPoint,
        QuadricPoint
    };
    struct alignas(16) OrientedPoint {
        // QUadric version
        float c{};
        float threshold{};
        float beta{};
        float _pad0{}; // align next float2 to 16‑byte boundary
        uint32_t material{};
        uint32_t _pad1{}; // 32 B
        // 2DGS
        float opacity;
        float3 color{};
        float covX;
        float covY;
        PointType type = QuadricPoint;
        uint32_t _pad2{}; // 32 B
    };

    CHECK_16(OrientedPoint);

    struct alignas(16) BVHNode {
        float3 aabbMin; // 16
        float3 aabbMax; // 32
        uint32_t leftFirst{}; // 36
        uint32_t triCount{}; // 40
        bool isLeaf() const {
            return triCount > 0;
        }
    };

    CHECK_16(BVHNode);

    struct alignas(16) BLASRange {
        uint32_t firstNode{};
        uint32_t nodeCount{};
        uint32_t _pad0{};
        uint32_t _pad1{}; // 16
    };

    CHECK_16(BLASRange);

    struct alignas(16) TLASNode {
        float3 aabbMin; // 16
        float3 aabbMax; // 32
        uint32_t leftChild{}; // 36
        uint32_t count{}; // 40
        uint32_t rightChild{}; // 44
        uint32_t _pad0{}; // 48
    };

    CHECK_16(TLASNode);

    /*************************  Appearance ***************************/
    struct alignas(16) Material {
        float baseColor{};
        float diffuse{};
        float specular{};
        float phongExp{}; // 16
    };

    CHECK_16(Material);

    /*************************  Transform ****************************/
    struct alignas(16) Transform {
        float4x4 objectToWorld{}; //  64
        float4x4 worldToObject{}; // 128
    };

    CHECK_16(Transform);

    /*************************  Mesh‑area Lights *********************/
    struct alignas(16) MeshLight {
        static constexpr size_t MAX_TRIANGLES = 64;

        float3 v0[MAX_TRIANGLES];
        float3 edge1[MAX_TRIANGLES];
        float3 edge2[MAX_TRIANGLES];
        float3 normal[MAX_TRIANGLES];
        float cdf[MAX_TRIANGLES]{};

        uint32_t triangleCount{0};
        float totalArea{0.f};
        float flux{0.f};
        float radiance{0.f};

        Transform transform{};

        void addTriangle(const float3 &p0,
                         const float3 &e1,
                         const float3 &e2,
                         const float3 &n,
                         float area) {
            if (triangleCount >= MAX_TRIANGLES)
                return; // (host version throws; device just skips)

            v0[triangleCount] = p0;
            edge1[triangleCount] = e1;
            edge2[triangleCount] = e2;
            normal[triangleCount] = n;

            totalArea += area;
            cdf[triangleCount] = totalArea;
            ++triangleCount;
        }

        void finalize() {
            if (totalArea == 0.f) return;
            for (uint32_t i = 0; i < triangleCount; ++i)
                cdf[i] /= totalArea;

            radiance = flux / (M_PI * totalArea);
        }
    };

    CHECK_16(MeshLight);

    /*************************  Scene graph **************************/
    enum class GeometryType : uint32_t { Mesh = 0, PointCloud = 1 };

    struct alignas(16) Instance {
        GeometryType geomType{GeometryType::Mesh};
        uint32_t geomIndex{0};
        uint32_t materialIndex{0};
        uint32_t transformIndex{0};
    };

    CHECK_16(Instance);

    struct alignas(16) MeshRange {
        uint32_t firstTri{}, triCount{};
        uint32_t firstVert{}, vertCount{}; // 16
    };

    CHECK_16(MeshRange);

    struct alignas(16) PointCloudRange {
        uint32_t firstPoint{};
        uint32_t pointCount{};
        uint32_t _pad0{};
        uint32_t _pad1{}; // 16
    };

    CHECK_16(PointCloudRange);

    struct alignas(16) Camera {
        float4x4 view{}; //  64
        float4x4 proj{}; // 128
        float4x4 invView{}; //  64
        float4x4 invProj{}; // 128
        float3 pos{}; // 144
        float3 forward{}; // 160
        uint32_t width{}, height{}; // 168
        uint32_t firstPixel{}; // 172
        Entity entity;
    };

    CHECK_16(Camera);

    //--------------------------------------------------------------------
    // Parameter indexing (done once on the CPU before any kernel launch)
    //--------------------------------------------------------------------
    struct ParamOffset {
        uint32_t start;   // first slot in d_gradAll
        uint32_t dim;     // 1 for scalar, 3 for float3, ...
    };


    // Storage for the photon map
    struct Photon {
        float3 position;
        float power = -1.0f;
    };

    /*************************  Scene descriptor *********************/
    struct alignas(16) SceneDesc {
        /* geometry */
        const Triangle *triangles = nullptr;
        const MeshRange *meshes = nullptr;
        const OrientedPoint *points = nullptr;
        const PointCloudRange *pointClouds = nullptr;
        const Vertex *vertices = nullptr;

        /* graph  */
        const Instance *instances = nullptr;
        const Transform *transforms = nullptr;

        /* BVHs    */
        const BVHNode *blasNodes = nullptr;
        const BLASRange *blasRanges = nullptr;
        const TLASNode *tlasNodes = nullptr;
        const BVHNode *betaNodes = nullptr;
        const BLASRange *betaRanges = nullptr;
        const uint32_t *triPerm = nullptr;

        /* appearance */
        const Material *materials = nullptr;
        const MeshLight *lights = nullptr;

        /* cameras */
        const Camera *cameras = nullptr;

        /* Parameter Gradients */
        ParamOffset * kdOffset = nullptr;
        ParamOffset * vtxOffset = nullptr;
        float      * gradAll = nullptr;
        // -- Gradient Debug
        float* gradKdImage = nullptr;
        uint32_t gradDebugMaterialID = 0;

        // -- Photon maps
        Photon* photonHits = nullptr;
        uint32_t* photonMapHitCount= nullptr; // keep 4‑byte – pad below

        /* counts */
        uint32_t triCount{}, vertexCount{}, meshCount{};
        uint32_t pointCount{}, pointCloudCount{};
        uint32_t instanceCount{}, transformCount{};
        uint32_t materialCount{}, lightCount{};
        uint32_t cameraCount{};
        uint32_t photonCount = 0; // keep 4‑byte – pad below

    };

    CHECK_16(SceneDesc);

    /*************************  Misc *********************************/
    struct alignas(16) RenderSettings {
        uint32_t maxBounces{32};
        uint32_t iteration = 0;
        uint64_t photonCount{1000}; // 16
        int64_t randomSeed{-1};
    };

    CHECK_16(RenderSettings);

    struct alignas(16) FrameBuffer {
        float4 *memory{nullptr};
        uint32_t frameBufferSize{0};
        uint32_t _pad0{};
        uint64_t _pad1{}; // 16

        float* residuals = nullptr;
        uint32_t residualBufferSize{0};
        uint64_t _pad2{}; // 16

        Photon* photonHits = nullptr;
        uint32_t photonHitBufferSize = 0;
    };

    CHECK_16(FrameBuffer);

    /*************************  Ray & Hit *****************************/
    struct alignas(16) Ray {
        float3 origin; // 16
        float3 direction; // 32
    };

    CHECK_16(Ray);

    inline Ray makeRay(float3 origin, float3 dir) {
        Ray r;
        r.origin = origin;
        r.direction = dir;
        return r;
    }

    struct alignas(16) Hit {
        float t{FLT_MAX};
        float u{0.f}, v{0.f};
        float _pad0{}; // 16
        float3 hitPoint; // 32
        uint32_t primIdx{UINT32_MAX};
        uint32_t instIdx{UINT32_MAX};
        GeometryType geomType{GeometryType::Mesh};
        uint32_t _pad1{}; // 48
    };

    CHECK_16(Hit);



#undef CHECK_16
} // namespace VkRender::PathTracer

#endif //GPUDATATYPES_H
