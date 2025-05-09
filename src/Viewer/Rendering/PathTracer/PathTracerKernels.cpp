//
// Created by magnus on 5/3/25.
//


#include "Viewer/Rendering/PathTracer/PathTracerTypes.h"
#include "Viewer/Rendering/PathTracer/PathTracerKernels.h"
#include "Viewer/Rendering/PathTracer/KernelHelpers.h"

namespace VkRender::PathTracer {
// ── PathTracerMeshKernel.cpp ────────────────────────────────────────────────
// Returns the closest hit inside one mesh’s BLAS (object space)
bool PathTracerMeshKernel::intersectBLAS(const Ray &rayO,
                                         uint32_t geomIdx,
                                         Hit      &out) const
{
    const SceneDesc  &scene = *d_sceneDesc;

    /* 1.  Locate the sub‑tree that belongs to this mesh
           ────────────────────────────────────────────── */
    const BLASRange &br    = scene.blasRanges[geomIdx];
    const BVHNode   *nodes = scene.blasNodes   + br.firstNode; // root = nodes[0]
    const Triangle  *tris  = scene.triangles;
    const Vertex    *verts = scene.vertices;

    /* 2.  Standard iterative depth‑first traversal
           ────────────────────────────────────────────── */
    float  bestT   = std::numeric_limits<float>::infinity();
    bool   hitAny  = false;
    float3 invDir  = 1.f / rayO.direction;

    int stack[64];                   // enough for >4 billion triangles
    int sp = 0;
    stack[sp++] = 0;                 // root of this BLAS

    while (sp)
    {
        int nIdx          = stack[--sp];
        const BVHNode &N  = nodes[nIdx];

        float tEntry;
        if (!slabIntersectAABB(rayO, N, invDir, bestT, tEntry))
            continue;                // miss or farther than current best

        if (N.triCount == 0)         // ── internal ─────────────────────
        {
            /* Push children – right first so left is processed next.
               Children are stored immediately after the parent once
               we patched indices in buildBLASForAllMeshes().          */
            stack[sp++] = N.leftFirst + 1;   // right
            stack[sp++] = N.leftFirst;       // left
        }
        else                         // ── leaf ─────────────────────────
        {
            for (uint32_t i = 0; i < N.triCount; ++i)
            {
                uint32_t triIdx      = N.leftFirst + i;  // *global* index
                const Triangle &T    = tris[triIdx];

                const float3 A = verts[T.v0].pos;
                const float3 B = verts[T.v1].pos;
                const float3 C = verts[T.v2].pos;

                float t,u,v;
                if (intersectTriangle(rayO, A,B,C, t,u,v) && t < bestT)
                {
                    bestT      = t;
                    hitAny     = true;

                    out.t      = t;
                    out.u      = u;
                    out.v      = v;
                    out.primIdx= triIdx;     // global – good for shading
                }
            }
        }
    }

    return hitAny;
}


    bool PathTracerMeshKernel::intersectScene(const Ray &rayW, Hit *hit) const {
        const SceneDesc &scene = *d_sceneDesc;

        /* abort if scene is empty */
        if (scene.tlasNodeCount == 0) return false;

        const TLASNode  *tlas   = scene.tlasNodes;
        const Instance  *instances  = scene.instances;
        const Transform *xforms = scene.transforms;


        /* ------------------------------------------------------------------ */
        /* stack‑based depth‑first traversal                                   */
        /* ------------------------------------------------------------------ */
        float bestT  = std::numeric_limits<float>::infinity();
        bool  anyHit = false;
        float3 invDir = 1.f / rayW.direction;


        int stack[64];
        int sp = 0;
        stack[sp++] = 0; // root

        while (sp) {
            int nIdx = stack[--sp];
            const TLASNode& node = tlas[nIdx];

            float tEntry;
            if (!slabIntersectAABB(rayW, node, invDir, bestT, tEntry))
                continue;

            if (node.count==0)          // internal
            {
                stack[sp++] = node.rightChild;
                stack[sp++] = node.leftChild;
            }     // leaf – exactly one instance
            else {
                uint32_t  instID = node.leftChild;
                const Instance  &instance  = instances [instID];
                const Transform &transform = xforms[instance.transformIndex];

                Ray rayObject = toObjectSpace(rayW, transform);

                //uint32_t triIdx  = scene.triIndices[node.leftFirst + i];
                //const Triangle &T = tris[triIdx];
                //hit->primIdx     = triIdx;
                Hit local;
                if (intersectBLAS(rayObject, instance.geomIndex, local) && local.t < bestT)
                {
                    bestT        = local.t;
                    anyHit       = true;
                    hit->t        = bestT;
                    hit->u        = local.u;
                    hit->v        = local.v;
                    hit->primIdx  = local.primIdx;
                    hit->instIdx  = instID;
                    hit->hitPoint = toWorldPoint(rayObject.origin + bestT*rayObject.direction, transform);
                }
            }
        }
        return anyHit;
    }


    void PathTracerMeshKernel::castContributions(
        const float3 &hitPoint,
        const float &throughput) const {
        const auto &scene = *d_sceneDesc;
        constexpr float kEps = 1e-4f;

        for (uint32_t camID = 0; camID < scene.cameraCount; ++camID) {
            const Camera &cam = scene.cameras[camID];

            // 1) build a ray from surface to camera aperture
            float3 toAperture = cam.pos - hitPoint;
            float distToA = sycl::length(toAperture);
            float3 dirToA = toAperture / distToA;
            Ray contribRay = makeRay(hitPoint + dirToA * kEps, dirToA);

            // 2) occlusion check
            Hit shadow;
            if (intersectScene(contribRay, &shadow) && shadow.t < distToA - kEps)
                continue;

            // 3) project hitPoint into clip space
            float4 worldPos = float4(hitPoint, 1.f);
            float4 viewPos = cam.view * worldPos; // TODO matrix vector prod
            float4 clipPos = cam.proj * viewPos; // TODO matrix vector prod
            float invW = 1.f / clipPos.w();
            float2 ndc = {clipPos.x() * invW, clipPos.y() * invW};
            if (ndc.x() < -1.f || ndc.x() > 1.f || ndc.y() < -1.f || ndc.y() > 1.f)
                continue;

            // 4) NDC → pixel coords
            auto px = static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * cam.width);
            auto py = static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * cam.height);
            uint32_t idx = cam.firstPixel + py * cam.width + px;

            // 5) atomic add into global image buffer (float4 array)
            auto &pixel = d_framebuffer->memory[idx];
            sycl::atomic_ref<float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    r(pixel.x());
            sycl::atomic_ref<float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    g(pixel.y());
            sycl::atomic_ref<float,

                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    b(pixel.z());

            r += throughput;
            g += throughput;
            b += throughput;
        }
    }

    //------------------------------------------------------------------------------
    /// Traces a single photon: emit from a light, bounce
    /// through the scene, shade, and cast contribution rays.
    /// \param photonID  Unique photon identifier (also RNG seed).
    //------------------------------------------------------------------------------
    void PathTracerMeshKernel::traceOnePhoton(uint32_t photonID) const {
        const auto &scene = *d_sceneDesc;
        const auto &settings = d_sceneSettings;

        PCG32 rng;
        uint64_t seedState = (uint64_t(photonID) << 32);
        rng.seed(seedState, /*stream*/ 54u);
        // 1) sample light
        float3 worldPos, worldNormal;
        float pdf;
        uint32_t lightIdx = sycl::max((photonID % scene.lightCount) - 1.0, 0.0);
        auto light = scene.lights[lightIdx];
        sampleMeshLight(light, rng, worldPos, worldNormal, pdf);


        // 2) initial direction & throughput
        float3 rayDir = sampleCosineHemisphere(worldNormal, rng);
        float cosNL = sycl::max(sycl::dot(rayDir, worldNormal), 0.f);
        float throughput = scene.lights[photonID % scene.lightCount].radiance
                           * (cosNL / pdf);
        Ray worldRay = makeRay(worldPos, rayDir);

        // 3) bounce loop
        for (uint32_t bounce = 0; bounce < settings.maxBounces; ++bounce) {
            Hit hit;
            if (!intersectScene(worldRay, &hit))
                break;


            // interpolate normal from triangle
            const Triangle &T = scene.triangles[hit.primIdx];
            const auto &VSOA = scene.vertices;
            const Vertex &v0 = VSOA[T.v0];
            const Vertex &v1 = VSOA[T.v1];
            const Vertex &v2 = VSOA[T.v2];


            float w0 = 1.0f - hit.u - hit.v;
            float w1 = hit.u;
            float w2 = hit.v;
            float3 N = normalize(w0 * v0.norm + w1 * v1.norm + w2 * v2.norm);

            // fetch material
            const Instance &inst = scene.instances[hit.instIdx];
            const Material &M = scene.materials[inst.materialIndex];
            float3 albedo = M.baseColor;

            // throughput update (Lambertian)
            throughput *= albedo.x() * M_PIf;

            // cast contributions to cameras
            castContributions(hit.hitPoint, throughput);

            // spawn next bounce
            worldRay = spawnNextRay(hit, N, rng);
        }
    }
}
