//
// Created by magnus on 5/2/25.
//

#ifndef GPUDATATYPES_H
#define GPUDATATYPES_H

#include <cstdint>

#include "PathTracerTypes.h"

namespace VkRender {
    struct QuadricCloudAsset;
}

namespace VkRender::PathTracer {
    // helper

    /**** Core Data Types ****/

    struct alignas(16) Vertex {
        float3 pos;
        float3 norm;
        // add UVs, tangents, etc. here if you need
    };

    struct alignas(16) Triangle {
        uint32_t v0, v1, v2;   // indices into a std::vector<Vertex>
        float3 centroid;
    };


    struct alignas(16) OrientedPoint // your “beta‑kernel plane”
    {
        float3 pos;
        float3 normal;
        float c;
        float threshold; // controls kernel extent
        float beta;
        float2 minSupport;
        float2 maxSupport;

        uint32_t material; // who shades this point?

    };

    struct alignas(32) BVHNode {
        float3    aabbMin, aabbMax;
        uint32_t  leftFirst;   // if leaf: index of first triangle, else index of left child
        uint32_t  triCount;    // >0 => leaf; ==0 => internal
        bool      isLeaf() const { return triCount > 0; }
    };

    // A contiguous slice of the global blasNodes[] buffer
    struct alignas(16) BLASRange {
        uint32_t firstNode;   // index of the BLAS root inside blasNodes[]
        uint32_t nodeCount;   // how many nodes belong to this BLAS
    };

    struct alignas(16) TLASNode   // one node in the top‑level tree
    {
        float3    aabbMin, aabbMax;
        uint32_t  leftChild;   // index of child node  (same as BLAS: leaf vs internal rule)
        uint32_t  count;       // ==0 → internal, 1 → leaf that stores an instIdx
        uint32_t rightChild;
    };



    struct alignas(16) Material {
        float baseColor;
        float diffuse;
        float specular;
        float phongExp;
        // every material is usable by either mesh or point
    };

    struct alignas(16) Transform {
        float4x4 objectToWorld;
        float4x4 worldToObject;
    };


    struct alignas(16) MeshLight {
        static constexpr size_t MAX_TRIANGLES = 64;

        // Per-triangle data (fixed-size arrays)
        sycl::float3 v0[MAX_TRIANGLES];
        sycl::float3 edge1[MAX_TRIANGLES];
        sycl::float3 edge2[MAX_TRIANGLES];
        sycl::float3 normal[MAX_TRIANGLES];
        float cdf[MAX_TRIANGLES]; // prefix-sum(areas) normalized to [0,1]
        uint32_t triangleCount = 0;
        float totalArea = 0.0f; // sum of all triangle areas
        float flux = 0.0f; // Φ in watts
        float radiance = 0.0f; // L_e = Φ/(π*totalArea)

        Transform transform;

        // Add a new triangle to the light
        void addTriangle(const sycl::float3 &v0, const sycl::float3 &e1, const sycl::float3 &e2, const sycl::float3 &n, float area) {
            if (triangleCount >= MAX_TRIANGLES) {
                // Optionally log or assert if you expect this to be rare
                throw std::runtime_error("Too many triangles in the mesh Light");
            }

            this->v0[triangleCount] = v0;
            this->edge1[triangleCount] = e1;
            this->edge2[triangleCount] = e2;
            this->normal[triangleCount] = n;

            totalArea += area;
            cdf[triangleCount] = totalArea;
            triangleCount++;
        }

        // Finalize CDF and radiance after adding all triangles
        void finalize() {
            for (uint32_t i = 0; i < triangleCount; ++i) {
                cdf[i] /= totalArea;
            }
            radiance = flux / (M_PI * totalArea);
        }
    };

    /**** Scene graph layer ****/

    enum class GeometryType : uint32_t { Mesh = 0, PointCloud = 1 };

    struct alignas(16) Instance {
        uint32_t geomType = 0; // cast to GeometryType
        uint32_t geomIndex = 0; // which mesh or point cloud?
        uint32_t materialIndex = 0; // appearance
        uint32_t transformIndex = 0; // object→world
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



    struct alignas(16) Camera {
        float4x4 view{};
        float4x4 proj{};
        float3 pos{};
        float3 forward{}; // Sensor plane normal
        uint32_t width{}, height{};

        uint32_t firstPixel{}; // offset into a big framebuffer
    };


    struct alignas(16) SceneDesc {
        // geometry
        const Triangle *triangles = nullptr;
        const MeshRange *meshes = nullptr;
        const OrientedPoint *points = nullptr;
        const PointCloudRange *pointClouds = nullptr;
        const Vertex* vertices; // see §2

        // scene graph
        const Instance *instances = nullptr;
        const Transform *transforms = nullptr;

        // Bvh
        const BVHNode *blasNodes = nullptr;   // flattened storage for every BLAS node
        const BLASRange *blasRanges = nullptr;
        const TLASNode *tlasNodes = nullptr;
        uint32_t blasNodeCount = 0;
        uint32_t tlasNodeCount = 0;

        // appearance
        const Material *materials = nullptr;
        const MeshLight *lights = nullptr;

        // view
        const Camera *cameras = nullptr;

        // counts (uint32 keeps struct 16‑byte aligned)
        uint32_t triCount = 0, vertexCount = 0, meshCount = 0;
        uint32_t pointCount = 0, pointCloudCount = 0;
        uint32_t instanceCount = 0, transformCount = 0;
        uint32_t materialCount = 0, lightCount = 0;
        uint32_t cameraCount = 0;
        unsigned int photonCount = 0;
    };


    struct alignas(16) SceneSettings {
        uint32_t maxBounces = 32;
    };

    struct alignas(16) FrameBuffer {
        float4 *memory = nullptr;
        uint32_t frameBufferSize = 0;
    };
}

#endif //GPUDATATYPES_H
