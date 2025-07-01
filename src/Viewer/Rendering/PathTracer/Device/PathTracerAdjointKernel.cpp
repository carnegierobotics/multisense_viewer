//
// Created by magnus on 6/11/25.
//

#include "PathTracerAdjointKernel.h"

#include <Viewer/Rendering/PathTracer/PathTracerTypes.h>

namespace VkRender::PathTracer {
    SYCL_EXTERNAL void PathTracerAdjointKernel::traceAdjoint(int px, int py) const {
        const SceneDesc &scene = *d_sceneDesc;
        const RenderSettings &settings = d_sceneSettings;
        const Camera &camera = d_sceneDesc->cameras[1];
        const float eps = 1e-4f;

        /* RNG --------------------------------------------------------------------- */
        PCG32 rng{};
        rng.seed((d_sceneDesc->cameras[1].width * d_sceneDesc->cameras[1].height * settings.randomSeed) + (py * d_sceneDesc->cameras[1].width + px));

        // 1) Compute pixel index and residual δy at (px,py)
        uint32_t pixelID = py * d_sceneDesc->cameras[1].width + px;

        float residual = d_framebuffer.residuals[pixelID];

        float3 dirCam;
        float3 camOrigin;
        float pdfCam;
        sampleCameraRay(px, py, camera, rng, camOrigin, dirCam, pdfCam); //   W_k / pdf
        float adjWeight = residual / pdfCam; //   A_e   (scalar)

        Ray ray = makeRay(camOrigin, dirCam);

        for (int bounce = 0; bounce < settings.maxBounces; ++bounce) {
            Hit hit;
            if (!intersectScene(ray, &hit, scene)) break;

            const Instance &inst = scene.instances[hit.instIdx];
            const Transform &xfInst = scene.transforms[inst.transformIndex];
            const Material &mat = scene.materials[inst.materialIndex];

            float3 surfaceNormal; // will be set per-geometry
            float3 worldNormal; // will be set per-geometry
            if (inst.geomType == GeometryType::Mesh) {
                const Triangle &tri = scene.triangles[hit.primIdx];
                const Vertex &v0 = scene.vertices[tri.v0];
                const Vertex &v1 = scene.vertices[tri.v1];
                const Vertex &v2 = scene.vertices[tri.v2];

                // Interpolated normal
                float w0 = 1.f - hit.u - hit.v;
                float w1 = hit.u;
                float w2 = hit.v;
                surfaceNormal = normalize(w0 * v0.norm + w1 * v1.norm + w2 * v2.norm); // Local surface normal:
                worldNormal = transformNormal(surfaceNormal, xfInst.objectToWorld); // world-space

                // World surface normal:
                float3 P0_obj = v0.pos;
                float3 P1_obj = v1.pos;
                float3 P2_obj = v2.pos;
                float3 Ng_obj = normalize(cross(P1_obj - P0_obj, P2_obj - P0_obj));
                worldNormal = transformNormal(Ng_obj, xfInst.objectToWorld); // world-space
            }

            // ---------- 1)  incident radiance via photon map or NEE ---------
            float Li = 1.0f; // Biased gradient estimateIncidentRadiance(hit); // reuse your photon map

            // ---------- 2)  accumulate gradient for this material ----------
            const ParamOffset kd = scene.kdOffset[inst.materialIndex];
            float delta = (Li * adjWeight) / M_PIf;
            addToGradBuffer<1>( scene.gradAll, kd.start, { delta } );

            // NEW: write per-pixel gradient image, but only if this is obj-1
            if(inst.materialIndex == scene.gradDebugMaterialID) {
                uint32_t pxIdx = py * camera.width + px;   // 2-D → 1-D
                sycl::atomic_ref<float,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    cell( scene.gradKdImage[pxIdx] );
                cell += delta;                            // add Δ once per path
            }

            // ---------- 3)  propagate importance to next bounce ------------
            float3 newDir{};
            float pdfDir = 1.0f;
            sampleCosineHemisphere(rng, worldNormal, newDir, pdfDir);
            float cosNO = fabs(dot(worldNormal, newDir));
            float bsdfAdj = (mat.diffuse / M_PIf) * cosNO / pdfDir; // adjoint weight
            adjWeight *= bsdfAdj;

            // ---------- 4)  Russian-roulette etc. (as in forward pass) -----
            // … identical to your photon loop …

            ray = makeRay(hit.hitPoint + eps * newDir, newDir);
        }
    }

    // ── PathTracerAdjointKernel.cpp ────────────────────────────────────────────────
    // Returns the closest hit inside one mesh’s BLAS (object space)
    SYCL_EXTERNAL bool PathTracerAdjointKernel::intersectBLASMesh(const Ray &rayO,
                                                                  uint32_t geomIdx,
                                                                  Hit &out, const SceneDesc &scene) {
        /* 1.  Locate the sub‑tree that belongs to this mesh
               ────────────────────────────────────────────── */
        const BLASRange &br = scene.blasRanges[geomIdx];
        const BVHNode *nodes = scene.blasNodes + br.firstNode; // root = nodes[0]
        const Triangle *tris = scene.triangles;
        const Vertex *verts = scene.vertices;

        /* 2.  Standard iterative depth‑first traversal
               ────────────────────────────────────────────── */
        float bestT = std::numeric_limits<float>::infinity();
        bool hitAny = false;
        float3 invDir = safeInvDir(rayO.direction);

        SmallStack<4096> stack;
        stack.push(0); // root

        while (!stack.empty()) {
            int nIdx = stack.pop();

            const BVHNode &N = nodes[nIdx];

            float tEntry;
            if (!slabIntersectAABB(rayO, N, invDir, bestT, tEntry))
                continue; // miss or farther than current best

            if (N.triCount == 0) // ── internal ─────────────────────
            {
                /* Push children – right first so left is processed next.
                   Children are stored immediately after the parent once
                   we patched indices in buildBLASForAllMeshes().          */
                // when you push children:

                if (!stack.push(N.leftFirst + 1)) return hitAny; // overflow → miss
                if (!stack.push(N.leftFirst)) return hitAny;
            } else // ── leaf ─────────────────────────
            {
                for (uint32_t i = 0; i < N.triCount; ++i) {
                    uint32_t triIdx = N.leftFirst + i; // *global* index

                    const Triangle &T = tris[triIdx]; // existing line
                    const float3 A = verts[T.v0].pos;
                    const float3 B = verts[T.v1].pos;
                    const float3 C = verts[T.v2].pos;

                    float t = FLT_MAX;
                    float u = 0;
                    float v = 0;

                    /* ---- new call --------------------------------------------------------- */
                    if (intersectTriangle(rayO, A, B, C,
                                          t, u, v)
                        && t < bestT) {
                        bestT = t;
                        hitAny = true;

                        out.t = t;
                        out.u = u;
                        out.v = v;
                        out.primIdx = triIdx; // global – good for shading
                    }
                }
            }
        }

        return hitAny;
    }

    //---------------------------------------------------------------------
    // Returns the closest hit inside one *quadric-patch* BLAS (object space)
    //---------------------------------------------------------------------
    SYCL_EXTERNAL bool PathTracerAdjointKernel::intersectBLASQuadric(
        const Ray &rayO, // object-space ray
        uint32_t geomIdx, // which BLAS
        Hit &out, // result
        const SceneDesc &scene) {
        /* 1. locate the sub-tree that belongs to this point-cloud */
        const BLASRange &br = scene.betaRanges[geomIdx];
        const BVHNode *nodes = scene.betaNodes + br.firstNode; // root = nodes[0]
        const OrientedPoint *patch = scene.points; // global array

        /* 2. standard iterative depth-first traversal */
        float bestT = std::numeric_limits<float>::infinity();
        bool hitAny = false;
        float3 invDir = safeInvDir(rayO.direction);

        int stack[64] = {-1}; // first entry = −1, rest 0
        int sp = 0;
        stack[sp++] = 0; // push root

        constexpr float kRayEps = 0.0f; // ignore hits closer than this

        while (sp) {
            int nIdx = stack[--sp];
            const BVHNode &N = nodes[nIdx];

            float tEntry;
            if (!slabIntersectAABB(rayO, N, invDir, bestT, tEntry))
                continue; // miss or too far

            if (N.triCount == 0) // ── internal node ──────────────────
            {
                /* children are stored next to each other (leftFirst , +1)     */
                stack[sp++] = N.leftFirst + 1; // right
                stack[sp++] = N.leftFirst; // left  (processed next)
            } else // ── leaf with β-patches ────────────
            {
                for (uint32_t i = 0; i < N.triCount; ++i) {
                    uint32_t pIdx = N.leftFirst + i; // *global* patch index
                    const OrientedPoint &P = patch[pIdx];

                    bool reflected;
                    float t;
                    float kHit;
                    if (intersectPatch(rayO, P, t, kHit, reflected) && // exact quadric test
                        t > kRayEps && t < bestT && reflected) {
                        bestT = t;
                        hitAny = true;

                        out.t = t;
                        out.u = kHit; // steal any unused field
                        out.v = reflected ? 1.f : 0.f;
                        out.primIdx = pIdx; // for shading
                        out.geomType = GeometryType::PointCloud;
                    }
                }
            }
        }
        return hitAny;
    }


    SYCL_EXTERNAL bool PathTracerAdjointKernel::intersectScene(const Ray &rayW, Hit *hit, const SceneDesc &scene) {
        /* abort if scene is empty */
        //if (scene.tlasNodeCount == 0) return false;

        const TLASNode *tlas = scene.tlasNodes;
        const Instance *instances = scene.instances;
        const Transform *xforms = scene.transforms;

        /* ------------------------------------------------------------------ */
        /* stack‑based depth‑first traversal                                   */
        /* ------------------------------------------------------------------ */
        bool anyHit = false;
        float3 invDir = safeInvDir(rayW.direction);;

        SmallStack<> stack;
        stack.push(0); // root
        float bestTWorld = std::numeric_limits<float>::infinity();

        while (!stack.empty()) {
            int nIdx = stack.pop();
            const TLASNode &node = tlas[nIdx];


            float tEntry;
            if (!slabIntersectAABB(rayW, node, invDir, bestTWorld, tEntry))
                continue;

            if (node.count == 0) // internal
            {
                if (!stack.push(node.rightChild)) return false;
                if (!stack.push(node.leftChild)) return false;
            } // leaf – exactly one instance
            else {
                uint32_t instID = node.leftChild;
                const Instance &instance = instances[instID];
                const Transform &transform = xforms[instance.transformIndex];

                Ray rayObject = toObjectSpace(rayW, transform);

                Hit local;

                bool ok = false;
                if (instance.geomType == GeometryType::Mesh) {
                    ok = intersectBLASMesh(rayObject, instance.geomIndex, local, scene);
                } else {
                    // point cloud
                    ok = intersectBLASQuadric(rayObject, instance.geomIndex, local, scene);
                }
                if (ok) {
                    /* 3.  Convert hit point back to world space */
                    float3 hitPointW = toWorldPoint(rayObject.origin + local.t * rayObject.direction, transform);

                    /* 4.  Compute world‑space hit distance (rayW.dir is unit length) */
                    float tWorld = dot(hitPointW - rayW.origin, rayW.direction);

                    if (tWorld >= 0.0f && tWorld < bestTWorld) // ignore hits behind the origin
                    {
                        bestTWorld = tWorld;
                        anyHit = true;

                        /* write out the final hit record in WORLD space */
                        hit->geomType = local.geomType;
                        hit->t = tWorld;
                        hit->u = local.u;
                        hit->v = local.v;
                        hit->primIdx = local.primIdx;
                        hit->instIdx = instID;
                        hit->hitPoint = hitPointW;
                    }
                }
            }
        }
        return anyHit;
    }
}
