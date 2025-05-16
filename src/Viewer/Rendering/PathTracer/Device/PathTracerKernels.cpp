//
// Created by magnus on 5/3/25.
//


#include "Viewer/Rendering/PathTracer/PathTracerTypes.h"
#include "Viewer/Rendering/PathTracer/Device/PathTracerKernels.h"
#include "Viewer/Rendering/PathTracer/Device/KernelHelpers.h"

namespace VkRender::PathTracer {
    // ── PathTracerMeshKernel.cpp ────────────────────────────────────────────────
    // Returns the closest hit inside one mesh’s BLAS (object space)
    SYCL_EXTERNAL bool PathTracerMeshKernel::intersectBLAS(const Ray &rayO,
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

        int stack[64] = {-1}; // enough for >4 billion triangles
        int sp = 0;
        stack[sp++] = 0; // root of this BLAS

        constexpr float kRayEps = 0.0f; // ignore hits closer than this

        while (sp) {
            int nIdx = stack[--sp];
            const BVHNode &N = nodes[nIdx];

            float tEntry;
            if (!slabIntersectAABB(rayO, N, invDir, bestT, tEntry))
                continue; // miss or farther than current best

            if (N.triCount == 0) // ── internal ─────────────────────
            {
                /* Push children – right first so left is processed next.
                   Children are stored immediately after the parent once
                   we patched indices in buildBLASForAllMeshes().          */
                stack[sp++] = N.leftFirst + 1; // right
                stack[sp++] = N.leftFirst; // left
            } else // ── leaf ─────────────────────────
            {
                for (uint32_t i = 0; i < N.triCount; ++i) {
                    uint32_t triIdx = N.leftFirst + i; // *global* index

                    const Triangle &T = tris[triIdx]; // existing line
                    const float3 A = verts[T.v0].pos;
                    const float3 B = verts[T.v1].pos;
                    const float3 C = verts[T.v2].pos;

                    float t, u, v;

                    /* ---- new call --------------------------------------------------------- */
                    if (intersectTriangle(rayO, A, B, C,
                                          t, u, v,
                                          kRayEps, /* tMin                */
                                          false) /* double‑sided, i.e. culling */
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


    SYCL_EXTERNAL bool PathTracerMeshKernel::intersectScene(const Ray &rayW, Hit *hit, const SceneDesc &scene) {
        /* abort if scene is empty */
        if (scene.tlasNodeCount == 0) return false;

        const TLASNode *tlas = scene.tlasNodes;
        const Instance *instances = scene.instances;
        const Transform *xforms = scene.transforms;


        /* ------------------------------------------------------------------ */
        /* stack‑based depth‑first traversal                                   */
        /* ------------------------------------------------------------------ */
        bool anyHit = false;
        float3 invDir = safeInvDir(rayW.direction);;


        int stack[64];
        int sp = 0;
        stack[sp++] = 0; // root

        float bestTWorld = std::numeric_limits<float>::infinity();

        while (sp) {
            int nIdx = stack[--sp];
            const TLASNode &node = tlas[nIdx];

            float tEntry;
            if (!slabIntersectAABB(rayW, node, invDir, bestTWorld, tEntry))
                continue;

            if (node.count == 0) // internal
            {
                stack[sp++] = node.rightChild;
                stack[sp++] = node.leftChild;
            } // leaf – exactly one instance
            else {
                uint32_t instID = node.leftChild;
                const Instance &instance = instances[instID];
                const Transform &transform = xforms[instance.transformIndex];

                Ray rayObject = toObjectSpace(rayW, transform);

                Hit local;
                if (intersectBLAS(rayObject, instance.geomIndex, local, scene)) {
                    /* 3.  Convert hit point back to world space */
                    float3 hitPointW = toWorldPoint(rayObject.origin + local.t * rayObject.direction, transform);

                    /* 4.  Compute world‑space hit distance (rayW.dir is unit length) */
                    float tWorld = dot(hitPointW - rayW.origin, rayW.direction);

                    if (tWorld >= 0.0f && tWorld < bestTWorld) // ignore hits behind the origin
                    {
                        bestTWorld = tWorld;
                        anyHit = true;

                        /* write out the final hit record in WORLD space */
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


    SYCL_EXTERNAL void PathTracerMeshKernel::castContributions(
        const float3 &hitPoint,
        const float contrib) const {
        constexpr float kEps = 1e-5f;
        const SceneDesc &scene = *d_sceneDesc;

        for (uint32_t camID = 0; camID < scene.cameraCount; ++camID) {
            const Camera &cam = scene.cameras[camID];

            float3 toAperture = cam.pos - hitPoint;
            float distToA = sycl::length(toAperture);
            float3 dirToA = toAperture / distToA;

            // visibility
            Hit sh;
            Ray ray = makeRay(hitPoint + dirToA * kEps, dirToA);
            if (intersectScene(ray, &sh, scene) && sh.t < distToA - kEps)
                continue;

            // perspective projection
            float4 clip = cam.proj * (cam.view * float4(hitPoint, 1.f));
            float2 ndc = {clip.x() / clip.w(), clip.y() / clip.w()};
            if (ndc.x() < -1.f || ndc.x() > 1.f || ndc.y() < -1.f || ndc.y() > 1.f)
                continue;

            uint32_t px = uint32_t((ndc.x() * 0.5f + 0.5f) * cam.width);
            uint32_t py = uint32_t((ndc.y() * 0.5f + 0.5f) * cam.height);
            uint32_t idx = cam.firstPixel + py * cam.width + px;

            float4 &dst = d_framebuffer.memory[idx];
            sycl::atomic_ref<float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    r(dst.x()), g(dst.y()), b(dst.z());

            r += contrib;
            g += contrib;
            b += contrib;
        }
    }

    //------------------------------------------------------------------------------
    /// Traces a single photon: emit from a light, bounce
    /// through the scene, shade, and cast contribution rays.
    /// \param photonID  Unique photon identifier (also RNG seed).
    //------------------------------------------------------------------------------
    SYCL_EXTERNAL void PathTracerMeshKernel::traceOnePhoton(uint64_t photonID, uint32_t totalPhotonCount) const {
        const auto &scene = *d_sceneDesc;
        const auto &settings = d_sceneSettings;


        // RNG
        PCG32 rng{};
        rng.seed(static_cast<uint64_t>(photonID));

        //-------------------- 1) sample area light -------------------------------
        float3 Lpos, Lnorm;
        float pdfPos;
        uint32_t lightIdx = scene.lightCount - 1;
        const MeshLight &light = scene.lights[lightIdx];
        sampleMeshLight(light, rng, Lpos, Lnorm, pdfPos);

        //-------------------- 2) launch first ray --------------------------------
        float3 dir;
        float pdfDir;
        sampleCosineHemisphere(rng, Lnorm, dir, pdfDir);

        float cosNL = sycl::max(1e-5f, static_cast<float>(sycl::dot(dir, Lnorm)));
        //float cosNL = sycl::dot(dir, Lnorm);

        float perPhotonEnergy = 1.0f / static_cast<float>(totalPhotonCount);
        float throughput = light.radiance * cosNL / (pdfPos * pdfDir) * perPhotonEnergy;

        Ray ray = makeRay(Lpos, dir);

        //-------------------- 3) bounce loop -------------------------------------
        for (uint32_t bounce = 0; bounce < settings.maxBounces; ++bounce) {
            Hit hit;
            if (!intersectScene(ray, &hit, scene))
                break;

            const Triangle &tri = scene.triangles[hit.primIdx];
            const Vertex &v0 = scene.vertices[tri.v0];
            const Vertex &v1 = scene.vertices[tri.v1];
            const Vertex &v2 = scene.vertices[tri.v2];

            float w0 = 1.f - hit.u - hit.v;
            float w1 = hit.u;
            float w2 = hit.v;
            float3 N = normalize(w0 * v0.norm + w1 * v1.norm + w2 * v2.norm);

            size_t instanceID = scene.instances[hit.instIdx].materialIndex;
            const Material &mat = scene.materials[instanceID];

            // material parameters
            float kd = mat.diffuse * mat.baseColor;
            float ks = mat.specular; // specular weight 0..1
            float s = mat.phongExp; // Blinn exponent

            // sample direction  (keep your cosine-hemisphere sampler for now)
            float3 newDir;
            float pdfDir = 0.f;
            sampleCosineHemisphere(rng, N, newDir, pdfDir);
            const float cosNO = sycl::max(0.f, sycl::dot(N, newDir));
            // If cosNO < 1e-4 the exact ratio kd cancels numerically.
            // Clamp the ratio instead of the cosine:
            const float minCos = 1e-5f;
            const float brdfOverPdf =
                    (cosNO < minCos)
                        ? kd // just use the limit
                        : (kd / M_PIf) * cosNO / (cosNO * (1.0f / M_PIf)); // == kd

            throughput *= brdfOverPdf;

            //---------------- camera contributions -----------------------------
            castContributions(hit.hitPoint, throughput);


            //---------------- spawn next ray -----------------------------------
            ray = makeRay(hit.hitPoint + 1e-5f * newDir, newDir);
        }
    }
}
