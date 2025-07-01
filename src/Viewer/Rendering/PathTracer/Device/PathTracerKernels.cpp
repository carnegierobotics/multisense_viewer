//
// Created by magnus on 5/3/25.
//


#include "Viewer/Rendering/PathTracer/PathTracerTypes.h"
#include "Viewer/Rendering/PathTracer/Device/PathTracerKernels.h"
#include "Viewer/Rendering/PathTracer/Device/KernelHelpers.h"

namespace VkRender::PathTracer {

    // ── PathTracerMeshKernel.cpp ────────────────────────────────────────────────
    // Returns the closest hit inside one mesh’s BLAS (object space)
    SYCL_EXTERNAL bool PathTracerMeshKernel::intersectBLASMesh(const Ray &rayO,
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
    SYCL_EXTERNAL bool PathTracerMeshKernel::intersectBLASQuadric(
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


    SYCL_EXTERNAL bool PathTracerMeshKernel::intersectScene(const Ray &rayW, Hit *hit, const SceneDesc &scene) {
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

    //---------------------------------------------------------------------
    // Return   T  =  product( 1 − α_i )   along the segment [ray.origin .. ray.origin+tMax]
    //   • stops early if a fully opaque surface (α ≥ 1−1e−4) is hit
    //---------------------------------------------------------------------
    SYCL_EXTERNAL float PathTracerMeshKernel::traceVisibility(const Ray &rayIn,
                                                              float tMax,
                                                              const SceneDesc &scene,
                                                              PCG32 &rng) const {
        constexpr float kEps = 1e-4f; // step-off to avoid self-hits
        constexpr float kMinT = 1e-4f; // early-exit threshold

        float T = 1.f; // running transmittance
        Ray ray = rayIn; // will advance along the ray
        Hit h;

        while (true) {
            /* next surface along the ray ------------------------------------- */
            if (!intersectScene(ray, &h, scene) || h.t >= tMax)
                return T; // reached the camera

            /* hard, fully opaque geometry ------------------------------------ */
            if (h.geomType == GeometryType::Mesh)
                return 0.f; // blocked by triangle mesh

            /* semi-transparent point patch ----------------------------------- */
            const float alpha = h.u; //   α  = opacity  (stored in hit)

            /* accumulate deterministic transmittance T *= (1-α) */
            T *= (1.f - alpha);

            if (T < kMinT) // already almost black
                return 0.f;

            /* march on past this patch --------------------------------------- */
            float advance = h.t + kEps;
            ray.origin = ray.origin + advance * ray.direction;
            tMax -= advance;
        }
    }


    SYCL_EXTERNAL void PathTracerMeshKernel::castContributions(
        const Hit &hitPoint,
        float contrib,
        const float3 &surfaceNormal,
        PCG32 &rng) const {
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        const SceneDesc &scene = *d_sceneDesc;

        size_t instanceID = scene.instances[hitPoint.instIdx].materialIndex;
        const Material &mat = scene.materials[instanceID];
        // material parameters
        float kd = mat.diffuse * mat.baseColor;

        for (uint32_t camID = 0; camID < scene.cameraCount; ++camID) {
            const Camera &cam = scene.cameras[camID];

            float3 toAperture = cam.pos - hitPoint.hitPoint;
            float distToA = length(toAperture);
            float3 dirToA = toAperture / distToA;

            // visibility
            Ray visRay = makeRay(hitPoint.hitPoint + dirToA * kEps, dirToA);
            float Tvis = traceVisibility(visRay, distToA - kEps, scene, rng);
            if (Tvis <= 0.0f) continue; // completely blocked

            // BRDF to camera direction
            float brdf  = kd / M_PIf;                                  // ρ / π
            float contribCam = contrib * brdf;                   // Lambert


            // Attenuation (Geometry term)
            //float cosCam = sycl::fabs(dot(-dirToA, cam.forward)); // cam.normal = forward
            float surfaceCos   = sycl::fabs(dot(surfaceNormal, dirToA));
            float cameraCos    = sycl::fabs(dot(cam.forward, -dirToA));
            float G_cam = (surfaceCos * cameraCos) / (distToA * distToA);
            float weight = contribCam * G_cam * Tvis;


            // perspective projection
            float4 clip = cam.proj * (cam.view * float4(hitPoint.hitPoint, 1.f));

            /* 0)  reject anything that is on or behind the eye plane  */
            if (clip.w() <= 0.0f)   //  <── missing guard
                continue;

            /* 1)  NDC                                                 */
            float2 ndc = { clip.x() / clip.w(), clip.y() / clip.w() };
            if (ndc.x() < -1.f || ndc.x() > 1.f ||
                ndc.y() < -1.f || ndc.y() > 1.f)
                continue;

            /* 2)  raster coords (clamp to avoid the right/top fenceposts) */
            uint32_t px = sycl::clamp(
                static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * cam.width),
                0u, cam.width  - 1);
            uint32_t py = sycl::clamp(
                static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * cam.height),
                0u, cam.height - 1);


            uint32_t idx = cam.firstPixel + py * cam.width + px;

            float4 &dst = d_framebuffer.memory[idx];
            sycl::atomic_ref<float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    r(dst.x());

            r += weight;
            //g += weight;
            //b += weight;
        }
    }

    //------------------------------------------------------------------------------
    /// Traces a single photon: emit from a light, bounce
    /// through the scene, shade, and cast contribution rays.
    /// \param photonID  Unique photon identifier (also RNG seed).
    //------------------------------------------------------------------------------
    SYCL_EXTERNAL void PathTracerMeshKernel::traceOnePhoton(uint64_t photonID, uint32_t totalPhotonCount) const {
        /* aliases ---------------------------------------------------------------- */
        const SceneDesc &scene = *d_sceneDesc;
        const RenderSettings &settings = d_sceneSettings;

        float eps = 1e-4f;

        /* RNG --------------------------------------------------------------------- */
        PCG32 rng{};
        rng.seed(photonID * d_sceneSettings.randomSeed);

        //-------------------- 1) sample area light -------------------------------
        float3 Lpos{}, Lnorm{};
        float pdfPos = 0.0f;
        uint32_t lightIdx = scene.lightCount - 1;
        const MeshLight &light = scene.lights[lightIdx];
        sampleMeshLight(light, rng, Lpos, Lnorm, pdfPos);

        //-------------------- 2) launch first ray --------------------------------
        float3 dir{};
        float pdfDir = 0.0f;
        sampleCosineHemisphere(rng, Lnorm, dir, pdfDir);

        float cosNL = sycl::max(0.0f, static_cast<float>(dot(dir, Lnorm)));

        float perPhotonEnergy = 1.0f / static_cast<float>(totalPhotonCount);
        float throughput = light.radiance * cosNL / (pdfPos * pdfDir) * perPhotonEnergy;

        Ray ray = makeRay(Lpos, dir);

        //-------------------- 3) bounce loop -------------------------------------
        for (uint32_t bounce = 0; bounce < settings.maxBounces; ++bounce) {
            Hit hit;
            if (!intersectScene(ray, &hit, scene))
                break;


            /* ---- fetch instance + material once for both geom types ------------ */
            const Instance &inst = scene.instances[hit.instIdx];
            const Transform &xfInst = scene.transforms[inst.transformIndex];
            const Material &mat = scene.materials[inst.materialIndex];

            /* ---- compute world-space surface normal ---------------------------- */
            float3 surfaceNormal; // will be set per-geometry
            float3 worldNormal; // will be set per-geometry
            if (inst.geomType == GeometryType::Mesh) {
                /* ----- triangle path (unchanged) -------------------------------- */
                const Triangle &tri = scene.triangles[hit.primIdx];
                const Vertex &v0 = scene.vertices[tri.v0];
                const Vertex &v1 = scene.vertices[tri.v1];
                const Vertex &v2 = scene.vertices[tri.v2];

                // Interpolated normal
                float w0 = 1.f - hit.u - hit.v;
                float w1 = hit.u;
                float w2 = hit.v;
                surfaceNormal    =   normalize(w0 * v0.norm + w1 * v1.norm + w2 * v2.norm); // Local surface normal:
                worldNormal     = transformNormal(surfaceNormal, xfInst.objectToWorld);   // world-space

                // World surface normal:
                /* ---- (b)  geometric normal ---------------------------------------- */
                float3 P0_obj = v0.pos;
                float3 P1_obj = v1.pos;
                float3 P2_obj = v2.pos;
                float3 Ng_obj = normalize(cross(P1_obj - P0_obj, P2_obj - P0_obj));
                worldNormal     = transformNormal(Ng_obj, xfInst.objectToWorld);   // world-space




            } else /* ---------------- quadric patch ------------------------ */
            {

                const float alpha = hit.u;              // opacity
                const float tau   = 1.f - alpha;

                if (rng.nextFloat() >= alpha) {         // *** transmitted ***
                    throughput *= tau;                  // attenuate
                    ray.origin = hit.hitPoint + 1e-4f * ray.direction;
                    --bounce;                           // don’t consume a real bounce
                    continue;                           // trace further
                }


                // object-space normal is always (0,0,1); rotate to world space
                float3 nObj = float3(0.f, 0.f, 1.f);
                surfaceNormal = nObj;
            }

            // material parameters
            float kd = mat.diffuse * mat.baseColor;
            //---------------- camera contributions -----------------------------
            castContributions(hit, throughput, worldNormal, rng);

            // sample direction  (keep your cosine-hemisphere sampler for now)
            float3 newDir;
            float pdfDir = 0.f;
            sampleCosineHemisphere(rng, worldNormal, newDir, pdfDir);
            const float cosNO = sycl::max(0.f, dot(worldNormal, newDir));
            // If cosNO < 1e-4 the exact ratio kd cancels numerically.
            // Clamp the ratio instead of the cosine:
            const float minCos = 1e-5f;
            float brdfOverPdfDiffuse = 0.0f;
            if (cosNO < minCos) {
                brdfOverPdfDiffuse = kd;
            } else {
                brdfOverPdfDiffuse = (kd / M_PIf) * cosNO / pdfDir;
            }
            throughput *= brdfOverPdfDiffuse;


            // ---------- robust Russian-Roulette ----------|
            uint32_t minDepth = 3;
            float pLow = 0.2f;
            float pHigh = 0.90f;
            if (bounce >= minDepth) {
                // guard against NaN/inf or negative weights
                if (!sycl::isfinite(throughput) || throughput <= 0.0f)
                    break;
                float pSurvive = sycl::clamp(throughput, pLow, pHigh);
                if (rng.nextFloat() > pSurvive) // kill the path
                    break;
                throughput /= pSurvive; // keep estimator unbiased
            }

            //---------------- spawn next ray -----------------------------------
            ray = makeRay(hit.hitPoint + eps * newDir, newDir);
        }
    }
}
