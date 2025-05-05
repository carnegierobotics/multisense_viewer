//
// Created by magnus on 5/3/25.
//


#include "Viewer/Rendering/PathTracer/PathTracerTypes.h"
#include "Viewer/Rendering/PathTracer/PathTracerKernels.h"
#include "Viewer/Rendering/PathTracer/KernelHelpers.h"

namespace VkRender::PathTracer {

   bool PathTracerMeshKernel::intersectBLAS(
    const Ray &ray, uint32_t meshIdx, Hit &out) const
{
    // fetch the BLAS range for this mesh
    const auto &mr         = d_sceneDesc->meshes[meshIdx];
    const auto firstNode   = d_sceneDesc->blasRanges[meshIdx].firstNode;
    const auto *nodes      = d_sceneDesc->blasNodes;
    const auto *tris       = d_sceneDesc->triangles;
    const auto &vsoa       = d_sceneDesc->vertices;

    float bestT = std::numeric_limits<float>::infinity();
    bool  hitAny = false;
    float3 invDir = 1.f / ray.direction;

    // stack-based BVH traversal
    int stack[64], sp = 0;
    stack[sp++] = int(firstNode);

    while (sp) {
        int idx = stack[--sp];
        const auto &node = nodes[idx];

        float tEntry;
        if (!slabIntersectAABB(ray, node, invDir, bestT, tEntry))
            continue;

        if (node.count == 0) {
            // internal node → push children
            stack[sp++] = int(node.rightChild);
            stack[sp++] = int(node.leftChild);
        } else {
            // leaf: test each triangle
            for (uint32_t i = 0; i < node.count; ++i) {
                uint32_t triIdx = node.leftChild + i;  // global triangle index
                const auto &T  = tris[triIdx];
                float t,u,v;
                float3 A{ vsoa.px[T.v0], vsoa.py[T.v0], vsoa.pz[T.v0] };
                float3 B{ vsoa.px[T.v1], vsoa.py[T.v1], vsoa.pz[T.v1] };
                float3 C{ vsoa.px[T.v2], vsoa.py[T.v2], vsoa.pz[T.v2] };
                if (intersectTriangle(ray, A, B, C, t, u, v) && t < bestT) {
                    bestT  = t;
                    hitAny = true;
                    out.u       = u;
                    out.v       = v;
                    out.primIdx = triIdx;
                }
            }
        }
    }

    if (hitAny) {
        out.t = bestT;
        return true;
    }
    return false;
}


bool PathTracerMeshKernel::intersectScene(const Ray &rayW, Hit *hit) const
{
    const auto *tlas   = d_sceneDesc->tlas;
    const auto *insts  = d_sceneDesc->instances;
    const auto *xforms = d_sceneDesc->transforms;

    float bestT  = std::numeric_limits<float>::infinity();
    hit->t       = bestT;
    bool anyHit  = false;
    float3 invW  = 1.f / rayW.direction;

    // TLAS traversal stack
    int stack[64], sp = 0;
    stack[sp++] = 0;   // root node

    while (sp) {
        int idx = stack[--sp];
        const auto &node = tlas[idx];

        float tEntry;
        if (!slabIntersectAABB(rayW, node, invW, bestT, tEntry))
            continue;

        if (node.count == 0) {
            // internal → push children
            stack[sp++] = int(node.rightChild);
            stack[sp++] = int(node.leftChild);
        } else {
            // leaf: exactly one instance
            uint32_t instIdx = node.leftChild;
            const auto &inst = insts[instIdx];
            const auto &xf   = xforms[inst.transformIndex];

            // transform the world ray into object space
            Ray rayO;
            rayO.origin    = xf.worldToObject   * sycl::float4{rayW.origin, 1.f};
            rayO.direction = xf.worldToObject   * sycl::float4{rayW.direction, 0.f};

            // test against the mesh’s BLAS
            Hit local;
            if (intersectBLAS(rayO, inst.geomIndex, local) && local.t < bestT) {
                bestT       = local.t;
                anyHit      = true;
                hit->t      = bestT;
                hit->instIdx = instIdx;
                // world-space hit point
                float3 pO = rayO.origin + bestT * rayO.direction;
                float4 pW = xf.objectToWorld * sycl::float4{pO,1.f};
                hit->hitPoint = float3{pW.x(), pW.y(), pW.z()};
                hit->u       = local.u;
                hit->v       = local.v;
                hit->primIdx = local.primIdx;
            }
        }
    }

    return anyHit;
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
        if (intersectScene(contribRay, &shadow) && shadow.t < distToA - kEps)
            continue;

        // 3) project hitPoint into clip space
        float4 worldPos = float4(hitPoint, 1.f);
        float4 viewPos = cam.view * worldPos; // TODO matrix vector prod
        float4 clipPos = cam.proj * viewPos; // TODO matrix vector prod
        float  invW     = 1.f / clipPos.w();
        float2 ndc      = { clipPos.x()*invW, clipPos.y()*invW };
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

        // 1) sample light
        float3 pos, normal;
        float pdf;
        uint32_t lightIdx = sycl::max((photonID % scene.lightCount) - 1.0, 0.0);
        auto light = scene.lights[lightIdx];
        sampleMeshLight(light, photonID, pos, normal, pdf);

        // 2) initial direction & throughput
        float3 rayDir = sampleCosineHemisphere(normal, photonID ^ 0xC789, photonID ^ 0xD012);
        float cosNL = sycl::max(dot(rayDir, normal), 0.f);
        float throughput = scene.lights[photonID % scene.lightCount].radiance
                            * (cosNL / pdf);
        Ray ray = makeRay(pos, rayDir);

        // 3) bounce loop
        for (uint32_t bounce = 0; bounce < settings.maxBounces; ++bounce) {
            Hit hit;
            if (!intersectScene(ray, &hit))
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
