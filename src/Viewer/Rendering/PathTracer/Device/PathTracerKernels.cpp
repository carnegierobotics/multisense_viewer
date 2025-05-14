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
            const float   contrib) const
    {
        constexpr float kEps = 1e-4f;
        const SceneDesc &scene = *d_sceneDesc;

        for (uint32_t camID = 0; camID < scene.cameraCount; ++camID)
        {
            const Camera &cam = scene.cameras[camID];

            float3 toAperture = cam.pos - hitPoint;
            float  distToA    = sycl::length(toAperture);
            float3 dirToA     = toAperture / distToA;

            // visibility
            Hit sh;
            Ray ray = makeRay(hitPoint + dirToA * kEps, dirToA);
            if (intersectScene(ray, &sh, scene) && sh.t < distToA - kEps)
                continue;

            // perspective projection
            float4 clip = cam.proj * (cam.view * float4(hitPoint,1.f));
            float2 ndc  = { clip.x()/clip.w(), clip.y()/clip.w() };
            if (ndc.x() < -1.f || ndc.x() > 1.f || ndc.y() < -1.f || ndc.y() > 1.f)
                continue;

            uint32_t px = uint32_t((ndc.x()*0.5f+0.5f) * cam.width );
            uint32_t py = uint32_t((ndc.y()*0.5f+0.5f) * cam.height);
            uint32_t idx = cam.firstPixel + py*cam.width + px;

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
    SYCL_EXTERNAL void PathTracerMeshKernel::traceOnePhoton(uint32_t photonID) const {
        const auto &scene = *d_sceneDesc;
        const auto &settings = d_sceneSettings;


        // RNG
        PCG32 rng{};
        rng.seed(uint64_t(photonID) << 32, 54u);

        //-------------------- 1) sample area light -------------------------------
        float3 Lpos, Lnorm;
        float pdfPos;
        uint32_t lightIdx = sycl::min(photonID % scene.lightCount,
                                      scene.lightCount - 1u);
        const MeshLight &light = scene.lights[lightIdx];
        sampleMeshLight(light, rng, Lpos, Lnorm, pdfPos);

        //-------------------- 2) launch first ray --------------------------------
        float3 dir;
        float pdfDir;
        sampleCosineHemisphere(rng, Lnorm, dir, pdfDir);

        float cosNL = sycl::dot(dir, Lnorm);
        float throughput = light.radiance * (cosNL / (pdfPos * pdfDir));

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

            const Material &mat = scene.materials[
                scene.instances[hit.instIdx].materialIndex];

            // material parameters
            float kd = mat.baseColor.x(); // diffuse albedo
            float ks = mat.specular; // specular weight 0..1
            float s = mat.phongExp; // Blinn exponent

            //---------------- choose branch ------------------------------------
            float Pdiff = kd; // assume greyscale kd
            float Pspec = ks;
            float invSum = 1.f / sycl::max(1e-5f, Pdiff + Pspec);
            Pdiff *= invSum;
            Pspec *= invSum;

            float rBranch = rng.nextFloat();
            float3 newDir;
            float pdfDirSample;

            if (rBranch < Pdiff) // diffuse branch
            {
                sampleCosineHemisphere(rng, N, newDir, pdfDirSample);
                pdfDirSample *= Pdiff; // mixture pdf
            } else // specular branch
            {
                sampleBlinnPhongSpecular(rng, N, -ray.direction, s,
                                         newDir, pdfDirSample);
                pdfDirSample *= Pspec;
            }

            //---------------- BRDF value ----------------------------------------
            float f;
            float cosNO = sycl::max(0.f, sycl::dot(N, newDir));
            if (rBranch < Pdiff) // diffuse eval
                f = kd / M_PIf;
            else // specular eval
            {
                float3 h = normalize(newDir - ray.direction);
                float nh = sycl::max(0.f, sycl::dot(N, h));
                f = ks * (s + 2.f) * sycl::pow(nh, s) / (2.f * M_PIf);
            }

            throughput *= (f * cosNO) / sycl::max(1e-7f, pdfDirSample);

            /*
            //---------------- Russian roulette (optional) ----------------------
            if (bounce > 3) {
                float q = sycl::clamp(throughput * 0.9f, 0.f, 0.95f);
                if (rng.nextFloat() < q)
                    break;
                throughput /= (1.f - q);
            }
            */


            // camera direction and geometry factor
            float3 camDir   = normalize(scene.cameras[0].pos - hit.hitPoint);  // pin‑hole
            float  dist     = length(scene.cameras[0].pos - hit.hitPoint);
            float  cosNxCam  = sycl::max(0.f, sycl::dot(N, camDir));
            float  Gcam      = cosNxCam / (dist * dist);           // nA·(–ωA)=1 for pin‑hole

            // BRDF toward the camera  (reuse the same logic as for the bounce branch)
            float frCam;
            {
                float ks = mat.specular;
                float kd = mat.baseColor.x();
                if (rBranch < Pdiff)            // same branch test: just diff/spec flag
                    frCam = kd / M_PIf;
                else {
                    float3 h = normalize(camDir - ray.direction);
                    float  nh = sycl::max(0.f, sycl::dot(N, h));
                    frCam = ks * (s + 2.f) * sycl::pow(nh, s) / (2.f * M_PIf);
                }
            }

            // pixel filter: box  -->  ΔΩ_pix = A_pix / dist^2
            //float pixelSolid = scene.cameras[0].pixelArea / (dist * dist);
            float pixelSolid = 4.f / (scene.cameras[0].width * scene.cameras[0].height * 1.0f);          // 90° default

            // overall camera kernel
            float cameraWeight = frCam * Gcam * pixelSolid;
            float contrib      = throughput * cameraWeight;

            /*
            if (contrib > 0.1f) {
                break;
            }
            */
            //---------------- camera contributions -----------------------------
            castContributions(hit.hitPoint, contrib);

            //---------------- spawn next ray -----------------------------------
            ray = makeRay(hit.hitPoint + 1e-4f * newDir, newDir);
        }
    }
}
