//
// Created by magnus on 5/2/25.
//

#ifndef GPUDATATYPES_H
#define GPUDATATYPES_H

#include <cstdint>

namespace VkRender {
    struct QuadricCloudAsset;
}

namespace VkRender::PathTracer {
    // helper

    /**** Core Data Types ****/

    struct alignas(16) VertexSOA {
        const float *px = nullptr; // part of VertexSOA
        const float *py = nullptr;
        const float *pz = nullptr;
        const float *nx = nullptr;
        const float *ny = nullptr;
        const float *nz = nullptr;
    };

    struct alignas(16) Triangle {
        uint32_t v0, v1, v2; // indices into vertex SOA
    };

    struct alignas(16) OrientedPoint // your “beta‑kernel plane”
    {
        sycl::float3 pos;
        sycl::float3 normal;
        float radius; // controls kernel extent
        float beta;
        uint32_t material; // who shades this point?
    };

    struct alignas(16) Material {
        sycl::float3 baseColor;
        float pad0;
        sycl::float3 specular;
        float phongExp;
        sycl::float3 emissive;
        float pad1;
        // every material is usable by either mesh or point
    };

    struct alignas(16) AreaLight {
        sycl::float3 origin;
        float pad0;
        sycl::float3 edgeU;
        float pad1; // defines plane
        sycl::float3 edgeV;
        float pad2;
        sycl::float3 radiance;
        float area; // pre‑computed area = |U×V|
    };

    struct alignas(16) Transform {
        sycl::mfloat4 objectToWorld;
        sycl::mfloat4 worldToObject;
    };


    /**** Scene graph layer ****/

    enum class GeometryType : uint32_t { Mesh = 0, PointCloud = 1 };

    struct alignas(16) Instance {
        uint32_t geomType; // cast to GeometryType
        uint32_t geomIndex; // which mesh or point cloud?
        uint32_t materialIndex; // appearance
        uint32_t transformIndex; // object→world
    };

    // ‑‑Mesh‑‑
    struct alignas(16) MeshRange {
        uint32_t firstTri;
        uint32_t triCount;
        uint32_t firstVert;
        uint32_t vertCount;
    };

    // ‑‑Point cloud‑‑
    struct alignas(16) PointCloudRange {
        uint32_t firstPoint;
        uint32_t pointCount;
    };


    struct alignas(16) BVHNode {
        sycl::float3 bboxMin{};
        uint32_t leftFirst{}; // index of left child OR first prim
        sycl::float3 bboxMax{};
        uint32_t count{}; // 0 = inner, n>0 = leaf with n prims
    };

    struct alignas(16) Camera {
        sycl::mfloat4 view{};
        sycl::mfloat4 proj{};
        sycl::float3 pos{};
        float pad0{};
        uint32_t width{}, height{};
        uint32_t firstPixel{}; // offset into a big framebuffer
    };


    struct alignas(16) SceneDesc {
        // geometry
        const Triangle *triangles = nullptr;
        const MeshRange *meshes = nullptr;
        const OrientedPoint *points = nullptr;
        const PointCloudRange *pointClouds = nullptr;
        const BVHNode *bvh = nullptr;
        VertexSOA vertices; // see §2

        // scene graph
        const Instance *instances = nullptr;
        const Transform *transforms = nullptr;

        // appearance
        const Material *materials = nullptr;
        const AreaLight *lights = nullptr;

        // view
        const Camera *cameras = nullptr;

        // counts (uint32 keeps struct 16‑byte aligned)
        uint32_t triCount = 0, meshCount = 0;
        uint32_t pointCount = 0, pointCloudCount = 0;
        uint32_t instanceCount = 0, transformCount = 0;
        uint32_t materialCount = 0, lightCount = 0;
        uint32_t cameraCount = 0;
    };


    struct alignas(16) SceneSettings {
        uint32_t maxBounces = 32;
    };

    struct alignas(16) FrameBuffer {
        sycl::float4 *frameBuffers = nullptr;
        uint32_t frameBufferCount = 0;
        uint32_t perFrameBufferSize = 0;

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;

    };
}

#endif //GPUDATATYPES_H
