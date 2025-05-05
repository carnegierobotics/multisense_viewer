//
// Created by magnus on 5/3/25.
//


#include "PathTracerKernels.h"
#include "KernelHelpers.h"

namespace VkRender::PathTracer {

    bool PathTracerMeshKernel::intersectBVH(
        const Ray &ray,
        Hit *hit) const {
        if (!hit)
            return false;

        const auto &scene = *d_sceneDesc;
        const BVHNode *nodes = scene.bvh;
        const Triangle *tris = scene.triangles;
        const VertexSOA &vsoa = scene.vertices;

        float tMin = std::numeric_limits<float>::infinity();
        bool found = false;
        float bestU = 0.f, bestV = 0.f;
        float3 bestP{0.f, 0.f, 0.f};
        uint32_t bestPrim = 0, bestInst = 0;

        // Precompute inverse ray direction
        float3 invDir = {
            1.f / ray.direction.x(),
            1.f / ray.direction.y(),
            1.f / ray.direction.z()
        };

        // BVH stack
        constexpr int MAX_STACK = 64;
        int stack[MAX_STACK], sp = 0;
        stack[sp++] = 0;

        while (sp > 0) {
            int idx = stack[--sp];
            const auto &node = nodes[idx];

            float tEntry;
            if (!slabIntersectAABB(ray, node, invDir, tMin, tEntry))
                continue;

            if (node.count > 0) {
                // Leaf: test each triangle
                for (uint32_t i = node.leftFirst; i < node.leftFirst + node.count; ++i) {
                    float t, u, v;
                    float3 v0 = {vsoa.px[tris[i].v0], vsoa.py[tris[i].v0], vsoa.pz[tris[i].v0]};
                    float3 v1 = {vsoa.px[tris[i].v1], vsoa.py[tris[i].v1], vsoa.pz[tris[i].v1]};
                    float3 v2 = {vsoa.px[tris[i].v2], vsoa.py[tris[i].v2], vsoa.pz[tris[i].v2]};
                    if (intersectTriangle(ray, v0, v1, v2, t, u, v) && t < tMin) {
                        tMin = t;
                        bestU = u;
                        bestV = v;
                        bestPrim = i;
                        bestInst = 0;
                        bestP = ray.origin + ray.direction * t;
                        found = true;
                    }
                }
            } else {
                // Internal: push children
                int lc = node.leftFirst;
                int rc = lc + 1;
                if (sp + 2 <= MAX_STACK) {
                    stack[sp++] = rc;
                    stack[sp++] = lc;
                }
            }
        }

        if (found) {
            hit->t = tMin;
            hit->u = bestU;
            hit->v = bestV;
            hit->pad0 = 0.f;
            hit->hitPoint = bestP;
            hit->primIdx = bestPrim;
            hit->instIdx = bestInst;
            hit->pad1 = hit->pad2 = 0u;
            return true;
        }

        return false;
    }

    void PathTracerMeshKernel::castContributions(
    const float3 &hitPoint,
    const float &throughput) const
{
    const auto &scene = *d_sceneDesc;
    constexpr float kEps = 1e-4f;

    for (uint32_t camID = 0; camID < scene.cameraCount; ++camID) {
        const Camera &cam = scene.cameras[camID];

        // 1) build a ray from surface to camera aperture
        float3 toAperture = cam.pos - hitPoint;
        float  distToA    = sycl::length(toAperture);
        float3 dirToA     = toAperture / distToA;
        Ray   contribRay = makeRay(hitPoint + dirToA*kEps, dirToA);

        // 2) occlusion check
        Hit shadow;
        if (intersectBVH(contribRay, &shadow) && shadow.t < distToA - kEps)
            continue;

        // 3) project hitPoint into clip space
        float4 worldPos = float4(hitPoint, 1.f);
        float4 viewPos;// = cam.view * worldPos; // TODO matrix vector prod
        float4 clipPos;// = cam.proj * viewPos; // TODO matrix vector prod
        float  invW     = 1.f / clipPos.w();
        float2 ndc      = { clipPos.x()*invW, clipPos.y()*invW };
        if (ndc.x() < -1.f || ndc.x() > 1.f || ndc.y() < -1.f || ndc.y() > 1.f)
            continue;

        // 4) NDC → pixel coords
        auto px = static_cast<uint32_t>((ndc.x() * 0.5f + 0.5f) * cam.width);
        auto py = static_cast<uint32_t>((ndc.y() * 0.5f + 0.5f) * cam.height);
        uint32_t idx = cam.firstPixel + py * cam.width + px;

        // 5) atomic add into global image buffer (float4 array)
        auto &pixel = d_frameBuffer->memory[idx];
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

        // 1) sample light
        float3 pos, normal;
        float pdf;
        sampleMeshLight(scene.lights[static_cast<size_t>(rnd(photonID) * scene.lightCount)], photonID,
                        pos, normal, pdf);

        // 2) initial direction & throughput
        float3 rayDir = sampleCosineHemisphere(normal, photonID ^ 0xC789, photonID ^ 0xD012);
        float cosNL = sycl::max(dot(rayDir, normal), 0.f);
        float throughput = scene.lights[photonID % scene.lightCount].radiance
                            * (cosNL / pdf);
        Ray ray = makeRay(pos, rayDir);

        // 3) bounce loop
        for (uint32_t bounce = 0; bounce < settings.maxBounces; ++bounce) {
            Hit hit;
            if (!intersectBVH(ray, &hit))
                break;

            // interpolate normal from triangle
            const Triangle &T = scene.triangles[hit.primIdx];
            const auto &VSOA = scene.vertices;
            float3 n0 = {VSOA.nx[T.v0], VSOA.ny[T.v0], VSOA.nz[T.v0]};
            float3 n1 = {VSOA.nx[T.v1], VSOA.ny[T.v1], VSOA.nz[T.v1]};
            float3 n2 = {VSOA.nx[T.v2], VSOA.ny[T.v2], VSOA.nz[T.v2]};
            float3 N = normalize((1 - hit.u - hit.v) * n0 + hit.u * n1 + hit.v * n2);

            // fetch material
            const Instance &inst = scene.instances[hit.instIdx];
            const Material &M = scene.materials[inst.materialIndex];
            float3 albedo = M.baseColor;

            // throughput update (Lambertian)
            throughput *= albedo.x() * M_PI;

            // cast contributions to cameras
            castContributions(hit.hitPoint, throughput);

            // spawn next bounce
            ray = spawnNextRay(hit, N, photonID + bounce * 13, photonID + bounce * 17);
        }
    }
}
