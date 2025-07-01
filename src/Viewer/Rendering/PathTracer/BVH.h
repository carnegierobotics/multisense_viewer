//
// Created by magnus on 5/6/25.
//

#ifndef BVH_H
#define BVH_H
#include <cstdint>

#include "Viewer/Rendering/PathTracer/GPUDataTypes.h"
#include "Viewer/Rendering/PathTracer/PathTracerTypes.h"

namespace VkRender::PathTracer {
    //------------------------------------------------------------------------------
    // Basic BVH interface
    //------------------------------------------------------------------------------
    class BasicBVH {
    public:
        /// Build over 'tris' & 'verts'.
        static void build(std::vector<Triangle> &inTris,
                          const std::vector<Vertex> &verts,
                          std::vector<BVHNode> &nodes,
                          std::vector<uint32_t> &triIndices,
                          uint32_t maxLeafSize = 16) {
            // 1) copy and centroid
            std::vector<Triangle> &tris = inTris;
            // 2) init root
            nodes.clear();
            nodes.reserve(tris.size() * 2);
            triIndices.resize(tris.size());
            for (uint32_t i = 0; i < triIndices.size(); ++i)
                triIndices[i] = i;
            nodes.emplace_back();
            nodes[0].leftFirst = 0;
            nodes[0].triCount = uint32_t(tris.size());
            updateBounds(tris, verts, triIndices, nodes[0]);
            // 3) recursive split
            subdivide(tris, verts, nodes, triIndices, 0, maxLeafSize);
        }

    private:
        // (b) fit node.aabb to its triangles
        static void updateBounds(const std::vector<Triangle> &tris,
                                 const std::vector<Vertex> &verts,
                                 const std::vector<uint32_t> &triIndices,
                                 BVHNode &node) {
            float3 bmin{std::numeric_limits<float>::infinity()};
            float3 bmax{-std::numeric_limits<float>::infinity()};
            for (uint32_t i = 0; i < node.triCount; ++i) {
                const Triangle &T = tris[triIndices[node.leftFirst + i]];
                const float3 &p0 = verts[T.v0].pos;
                const float3 &p1 = verts[T.v1].pos;
                const float3 &p2 = verts[T.v2].pos;
                bmin = min(bmin, p0);
                bmin = min(bmin, p1);
                bmin = min(bmin, p2);
                bmax = max(bmax, p0);
                bmax = max(bmax, p1);
                bmax = max(bmax, p2);
            }
            node.aabbMin = bmin;
            node.aabbMax = bmax;
        }

        // (c) recursive subdivision
        static void subdivide(const std::vector<Triangle> &tris,
                              const std::vector<Vertex> &verts,
                              std::vector<BVHNode> &nodes,
                              std::vector<uint32_t> &triIndices,
                              uint32_t nodeIdx,
                              uint32_t maxLeafSize) {
            BVHNode &node = nodes[nodeIdx];
            if (node.triCount <= maxLeafSize)
                return;

            // 1) pick split axis = longest extent
            float3 extent = node.aabbMax - node.aabbMin;
            int axis = (extent.y() > extent.x() && extent.y() > extent.z())
                           ? 1
                           : (extent.z() > extent.x() && extent.z() > extent.y())
                                 ? 2
                                 : 0;

            // 2) find median by centroid along that axis
            auto start = triIndices.begin() + node.leftFirst;
            auto end = start + node.triCount;
            auto mid = start + (node.triCount >> 1);

            std::nth_element(start, mid, end,
                             [&](uint32_t a, uint32_t b) {
                                 return tris[a].centroid[axis] < tris[b].centroid[axis];
                             });

            uint32_t leftCount = uint32_t(mid - start);
            // 2b) if that was degenerate (all centroids equal), bail out
            if (leftCount == 0 || leftCount == node.triCount)
                return;

            // 3) carve out two new nodes
            uint32_t leftChild = uint32_t(nodes.size());
            uint32_t rightChild = leftChild + 1;
            nodes.emplace_back(); // for left
            nodes.emplace_back(); // for right

            // 4) assign their ranges
            nodes[leftChild].leftFirst = node.leftFirst;
            nodes[leftChild].triCount = leftCount;
            nodes[rightChild].leftFirst = node.leftFirst + leftCount;
            nodes[rightChild].triCount = node.triCount - leftCount;

            // 5) flip this into an internal node
            node.leftFirst = leftChild;
            node.triCount = 0;

            // 6) refit their AABBs
            updateBounds(tris, verts, triIndices, nodes[leftChild]);
            updateBounds(tris, verts, triIndices, nodes[rightChild]);

            // 7) recurse
            subdivide(tris, verts, nodes, triIndices, leftChild, maxLeafSize);
            subdivide(tris, verts, nodes, triIndices, rightChild, maxLeafSize);
        }
    };

    inline void buildOrthonormalBasis(const float3 &n, float3 &u, float3 &v) {
        // pick the axis least parallel to n
        float3 a = (fabs(n.x()) > 0.9f) ? float3(0, 1, 0) : float3(1, 0, 0);
        u = ::normalize(cross(a, n));
        v = cross(n, u); // already normalized
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // QuadricBVH2D.h – BVH for z-up beta patches using min/max extents
    // ─────────────────────────────────────────────────────────────────────────────
    class QuadricBVH2D {
    public:
        static void build(const std::vector<OrientedPoint> &pts,
                          std::vector<BVHNode> &nodes,
                          std::vector<uint32_t> &permutation,
                          uint32_t maxLeaf = 8) {
            const uint32_t N = uint32_t(pts.size());
            permutation.resize(N);
            std::iota(permutation.begin(), permutation.end(), 0u);

            nodes.clear();
            nodes.reserve(N * 2);
            nodes.emplace_back(); // root
            nodes[0].leftFirst = 0;
            nodes[0].triCount = N;

            updateBounds(pts, permutation, nodes[0]);
            subdivide(pts, nodes, permutation, 0, maxLeaf);
        }

    private:
        // helper – identical math you used while deciding keepVertex
        static float radiusFromKernel(float kernelScale,
                                      float beta,
                                      float threshold) {
            const float p = 4.0f * sycl::exp(beta); // 4 e^β
            const float rSq = 1.0f - sycl::pow(threshold, 1.0f / p);
            return kernelScale * sycl::sqrt(sycl::max(rSq, 0.f));
        }

        /* --------------------------------------------------------------------- */
        static constexpr float eps = 1.0e-4f; // half-thickness along z

        static void updateBounds(const std::vector<OrientedPoint> &pts,
                                 const std::vector<uint32_t> &perm,
                                 BVHNode &n) {
            float3 bmin(+FLT_MAX), bmax(-FLT_MAX);
            constexpr float eps = 1.0e-4f; // thin slab around z=0

            for (uint32_t i = 0; i < n.triCount; ++i) {
                const OrientedPoint &P = pts[perm[n.leftFirst + i]];

                /* 2. scale canonical rectangle by that radius            */
                float3 pMin, pMax;
                switch (P.type) {
                    case Gaussian2DPoint: {
                        // 2-sigma ellipse: cover ±2·σx in X, ±2·σy in Y
                        float rx = 2.0f * P.covX;
                        float ry = 2.0f * P.covY;
                        pMin = float3{-rx, -ry, -eps};
                        pMax = float3{ rx,  ry,  eps};
                        break;
                    }

                    case QuadricPoint: {
                        // spherical support as before
                        float R = radiusFromKernel(1.0f, P.beta, P.threshold);
                        pMin = float3{-R, -R, -eps};
                        pMax = float3{ R,  R,  eps};
                        break;
                    }
                }

                bmin = min(bmin, pMin);
                bmax = max(bmax, pMax);
            }
            n.aabbMin = bmin;
            n.aabbMax = bmax;
        }

        //---------------------------------------------------------------------
        //  Compute the patch’s axis-aligned centroid for splitting
        //---------------------------------------------------------------------
        static float3 centroid(const OrientedPoint &P) {
            switch (P.type) {
                case Gaussian2DPoint: {
                    // 2-sigma ellipse in X and Y
                    float rx = 2.0f * P.covX;
                    float ry = 2.0f * P.covY;
                    // average of [-rx,rx] × [-ry,ry] is (0,0)
                    // If your OrientedPoint also carries a translation/orientation,
                    // multiply by its frame here.'
                    return float3{0.0f, 0.0f, 0.0f};
                }

                case QuadricPoint: {
                    // spherical support
                    float R = radiusFromKernel(1.0f, P.beta, P.threshold);
                    // average of [-R,R] is 0
                    return float3{0.0f, 0.0f, 0.0f};
                }
            }
            // fallback—shouldn’t happen
            return float3{0.0f, 0.0f, 0.0f};
        }

        static void subdivide(const std::vector<OrientedPoint> &pts,
                              std::vector<BVHNode> &nodes,
                              std::vector<uint32_t> &perm,
                              uint32_t nodeIdx,
                              uint32_t maxLeaf) {
            BVHNode &node = nodes[nodeIdx];
            if (node.triCount <= maxLeaf) return;

            /* longest axis of this node’s extents */
            float3 ext = node.aabbMax - node.aabbMin;
            int axis = (ext.y() > ext.x() && ext.y() > ext.z()) ? 1 : (ext.z() > ext.x() && ext.z() > ext.y()) ? 2 : 0;

            auto S = perm.begin() + node.leftFirst,
                    E = S + node.triCount,
                    M = S + (node.triCount >> 1);

            std::nth_element(S, M, E,
                             [&](uint32_t a, uint32_t b) {
                                 return centroid(pts[a])[axis] < centroid(pts[b])[axis];
                             });

            uint32_t leftCnt = uint32_t(M - S);
            if (leftCnt == 0 || leftCnt == node.triCount) return; // degenerate

            uint32_t L = uint32_t(nodes.size()), R = L + 1;
            nodes.emplace_back();
            nodes.emplace_back();

            nodes[L].leftFirst = node.leftFirst;
            nodes[L].triCount = leftCnt;
            nodes[R].leftFirst = node.leftFirst + leftCnt;
            nodes[R].triCount = node.triCount - leftCnt;

            node.leftFirst = L;
            node.triCount = 0; // internal

            updateBounds(pts, perm, nodes[L]);
            updateBounds(pts, perm, nodes[R]);
            subdivide(pts, nodes, perm, L, maxLeaf);
            subdivide(pts, nodes, perm, R, maxLeaf);
        }
    };
} // VkRender

#endif //BVH_H
