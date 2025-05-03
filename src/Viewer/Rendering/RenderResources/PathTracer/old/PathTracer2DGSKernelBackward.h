//
// Created by magnus on 1/28/25.
//

#ifndef PATHTRACER2DGSKERNELBACKWARD_H
#define PATHTRACER2DGSKERNELBACKWARD_H

#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracerKernelCommon.h"

namespace VkRender::PathTracer {
    class LightTracerKernelBackward {
    public:
        LightTracerKernelBackward(GPUData gpuData,
                                  GPUDataOutput* gpuDataOutput,
                                  PCG32* rng)
            : m_gpuData(gpuData), m_gpuDataOutput(gpuDataOutput), m_rng(rng) {
            m_cameraTransform = m_gpuData.cameraTransform;
            m_camera = m_gpuData.pinholeCamera;
        }

        void operator()(sycl::item<1> item) const {
            size_t photonID = item.get_linear_id();
            if (photonID >= m_gpuData.renderInformation->totalPhotons) {
                return;
            }
            // Each thread traces one photon.
            //traceOnePhotonDirectLighting(photonID);
            //traceOnePhotonSingleBounceEmissiveGradient(photonID);
            traceOnePhotonSingleBounceObjectGradient(photonID);
            //traceOnePhotonSecondBounceObjectGradient(photonID);
        }

    private:
        GPUData m_gpuData{};
        GPUDataOutput* m_gpuDataOutput{};

        PCG32* m_rng;
        TransformComponent* m_cameraTransform{};
        PinholeCamera* m_camera{};

        /*
                // ---------------------------------------------------------
                // Second-Bounce Photon Trace (Multi-Bounce)
                // ---------------------------------------------------------
                void traceOnePhotonSecondBounceObjectGradient(size_t photonID) const {
                    GPUDataOutput::Bounce &object = m_gpuDataOutput[photonID].bounce[1];
                    size_t hitObjectID = object.quadricID;
                    if (hitObjectID > m_gpuData.numQuadrics) {
                        return;
                    }

                    if (!object.hitCamera)
                        return;
                    glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
                    glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;

                    auto &quadric = m_gpuData.quadricInputAssembly[hitObjectID];
                    glm::vec3 q_c = quadric.transform.getPosition();
                    glm::mat3 world2Quadric = quadric.transform.getTransform();
                    glm::vec3 e_o_local = world2Quadric * (e_o - q_c);

                    glm::vec3 hit = object.hitPointWorld;
                    glm::vec3 hitNormal = object.hitNormalWorld;

                    glm::vec3 rayOrigin = object.outGoingOrigin;
                    glm::vec3 rayDir = object.outGoingDirection;

                    glm::vec3 a_d = object.apertureDirection;
                    glm::vec3 p = object.apertureHitPoint;
                    glm::vec3 p_c = object.cameraHitPointLocal;
                    glm::vec3 g_hit2 = object.hitPointWorld;


                    GPUDataOutput::Bounce &prevBounce = m_gpuDataOutput[photonID].bounce[0];

                    glm::vec3 prev_hit = prevBounce.hitPointWorld;
                    glm::vec3 prev_hitNormal = prevBounce.hitNormalWorld;
                    glm::vec3 prev_rayOrigin = prevBounce.outGoingOrigin;
                    glm::vec3 prev_rayDir = prevBounce.outGoingDirection;
                    glm::vec3 prev_a_d = prevBounce.apertureDirection;
                    glm::vec3 prev_p = prevBounce.apertureHitPoint;
                    glm::vec3 prev_p_c = prevBounce.cameraHitPointLocal;
                    glm::vec3 prev_g_hit2 = prevBounce.hitPointWorld;

                    glm::vec3 e_d_local = world2Quadric * e_d;
                }

        */
        // ---------------------------------------------------------
        // Single Photon Trace (Single-Bounce)
        // ---------------------------------------------------------
        void traceOnePhotonSingleBounceObjectGradient(size_t photonID) const {
            GPUDataOutput::Bounce& object = m_gpuDataOutput[photonID].bounce[0];
            size_t hitObjectID = object.quadricID;
            if (hitObjectID > m_gpuData.numQuadrics) {
                return;
            }

            // Early‑out if the camera was not hit
            if (!object.hitCamera) {
                return;
            }

            // Camera transforms etc.
            auto camera2World = m_cameraTransform->getTransform();
            glm::mat4 world2Camera_4x4 = glm::inverse(camera2World);
            glm::mat3 w2c = glm::mat3(world2Camera_4x4); // top‑left 3×3
            glm::mat3 c2w = glm::transpose(w2c); // since camera2World is orthonormal
            glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 pinholePosition = m_cameraTransform->getPosition();
            glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

            // Aperture center
            glm::vec3 a_c = m_cameraTransform->getPosition();
            // Scene (Gaussian) data
            size_t gaussianID = m_gpuDataOutput[photonID].gaussianID;
            glm::vec3 e_c = m_gpuData.gaussianInputAssembly[gaussianID].position; // emission center
            glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
            glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;
            glm::vec3 f = cameraPlanePointWorld; // “focal plane” point or just known plane
            glm::vec3 f_n = cameraNormal; // plane’s normal

            float u = object.pixelCoordinate.x;
            float v = object.pixelCoordinate.y;
            int uInt = (int)std::round(u);
            int vInt = (int)std::round(v);
            if (uInt < 0 || vInt < 0 ||
                uInt >= (int)m_camera->m_parameters.width ||
                vInt >= (int)m_camera->m_parameters.height) {
                return;
            }
            size_t pixelIndex = vInt * m_camera->m_parameters.width + uInt;


            // The quadric in question
            auto& quadric = m_gpuData.quadricInputAssembly[hitObjectID];
            glm::vec3 quadricNormalLocal(0.0f, 0.0f, 1.0f);
            // Extract the model matrix from your quadric transform.
            glm::mat4 modelMatrix = quadric.transform.getTransform();
            // When transforming normals, build the 3x3 normal matrix as the inverse transpose
            // of the upper-left 3x3 part of the model matrix.
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(modelMatrix)));
            // Transform the local normal into world space.
            glm::vec3 quadricNormalWorld = glm::normalize(normalMatrix * quadricNormalLocal);
            // Now you can compare with the camera normal.
            float facingCameraDot = glm::dot(quadricNormalWorld, -cameraNormal);
            float facingLightSourceDot = glm::dot(quadricNormalWorld, -e_d);
            switch (hitObjectID) {
            case 0:
                e_o = e_o;
                break;
            case 1:
                e_d = e_d;
                break;
            case 2:
                e_o = e_o;
                break;
            }
            //if (facingCameraDot <= 0.25f || facingLightSourceDot <= 0.25f) {
            //    return;
            //}
            // Grab data from the forward pass
            // The transform for this quadric
            auto& quadInfo = object.quadInfo;

            glm::vec3 o = quadInfo.localRayOrigin; // q_o
            glm::vec3 d = quadInfo.localRayDirection; // q_dir

            // world↔local rotation (orthonormal, det = +1)
            glm::mat3 R_q2w = glm::mat3(quadric.transform.getTransform());
            glm::mat3 R_w2q = glm::transpose(R_q2w); // faster than glm::inverse()

            // 2) Constants of the plane in *local* space --------------------------------
            const glm::vec3 n(0.0f, 0.0f, 1.0f); // unit normal  (local)
            const float tmp_denom = glm::dot(n, d); // n·d
            const float eps = 1e-6f; // degeneracy guard


            // 3)   ∂t_min / ∂q_c  = ( R_w2qᵀ · n ) / ( n·d )
            glm::vec3 d_tmin_qc = (glm::transpose(R_w2q) * n) / tmp_denom;

            // 4)   J_qhit_qc_l = –R_w2q  +  d  ⊗  (∂t_min/∂q_c)
            glm::mat3 J_qhit_qc_l = -R_w2q + glm::outerProduct(d, d_tmin_qc);


            // Unpack the remaining forward‐pass quantities:
            float g_d = quadInfo.geodesic;
            float x_local = quadInfo.hitLocal.x;
            float y_local = quadInfo.hitLocal.y;
            float z_local = quadInfo.hitLocal.z;
            float b_beta = quadric.b_beta;

            // Compute the constant factor
            float K = std::exp(b_beta);

            // First derivative (not used in the final override)
            // Override with your final formula:
            float p_tmp = 4.0f;
            float exponent = std::exp(b_beta);
            float term1 = 1.0f - (g_d * g_d);
            float d_beta_dgd =
                -(8.0f * std::pow(term1, (4.0f * exponent)) * exponent * g_d)
                / (term1) * 0.1f;

            // Compute ∇ₓ g_d  where g_d = ‖(x,y,z)‖
            float denom = std::sqrt(x_local * x_local + y_local * y_local + z_local * z_local);
            float dgd_dx = x_local / denom;
            float dgd_dy = y_local / denom;
            float dgd_dz = z_local / denom;

            // Chain‐rule: ∇ₓ β = dβ/dg_d * ∇ₓ g_d
            float d_beta_dx = d_beta_dgd * dgd_dx;
            float d_beta_dy = d_beta_dgd * dgd_dy;
            float d_beta_dz = d_beta_dgd * dgd_dz;

            // Pack it into a vector
            glm::vec3 J_beta_xyz(d_beta_dx, d_beta_dy, d_beta_dz);

            // Finally: J_{β,qc} = J_{β,xyz} · J_{xyz,qc}
            // In Python:  J_beta_qc = np.dot(J_beta_xyz, J_qhit_qc_l)
            // In C++/GLM we can do row‐vector*matrix via the transpose trick:
            glm::vec3 J_beta_qc = glm::transpose(J_qhit_qc_l) * J_beta_xyz;
            // For demonstration, put the 2D partial dβ/du, dβ/dv in gradientImageU, gradientImageV
            //m_gpuData.gradientImageU[pixelIndex] = J_beta_xy.x;
            //m_gpuData.gradientImageV[pixelIndex] = J_beta_xy.y;

            // Also store the 3D partial J_Iuv_qc, plus maybe the q_hit_world in the same mat3
            glm::mat3 tmp(0.0f);
            // First column = derivative
            tmp[0][0] = J_beta_qc.x;
            tmp[1][0] = J_beta_qc.y;
            tmp[2][0] = J_beta_qc.z;

            // Second column = q_hit_world
            tmp[0][1] = object.hitPointWorld.x;
            tmp[1][1] = object.hitPointWorld.y;
            tmp[2][1] = object.hitPointWorld.z;

            // Third column left empty or used as you wish
            tmp[0][2] = J_beta_xyz.x;
            tmp[1][2] = J_beta_xyz.y;
            tmp[2][2] = J_beta_xyz.z;

            // Store in GPU data
            m_gpuData.gradientPixelCoordinates[photonID] = glm::vec2(u, v);
            m_gpuData.gradientImagePerObject[pixelIndex] = static_cast<float>(hitObjectID);

            m_gpuData.photonIDGradient[photonID] = tmp;
        }

        /*
        bool castContributionRay(const glm::vec3 &directLightingOrigin, const glm::vec3 &cameraPlaneNormalWorld,
                                 float apertureRadius, size_t photonID, float photonFlux,
                                 glm::vec3 &directLightDir,
                                 glm::vec3 &apertureHitPoint,
                                 glm::vec3 &cameraHitPointLocal,
                                 glm::vec2 &pixelCoordinates,
                                 float &camera_t
        ) const {
            // Calculate direct lighting

            directLightDir = sampleDirectionTowardAperture(
                directLightingOrigin,
                m_cameraTransform->getPosition(), // center of aperture
                cameraPlaneNormalWorld, // might be -X if your camera faces X, or -Z, etc.
                apertureHitPoint,
                apertureRadius,
                photonID
            );

            // Early exist if we are hitting the camera plane from behind, this happens if the aperture direction and camera plane normal are parallell or within that quadrant
            float direction = glm::dot(directLightDir, cameraPlaneNormalWorld);
            if (direction >= 0.0f) {
                return false;
            }
            // Check if contribution ray intersects geometry

            // Create contribution Rays and trace towards the camera
            // Trace our contribution ray
            glm::vec3 camHit(0.0f);
            float incidentAngle = 0.0f;
            bool cameraHit = checkCameraPlaneIntersection(directLightingOrigin, directLightDir, camHit,
                                                          camera_t, incidentAngle);
            if (cameraHit) {
                float closest_t = FLT_MAX;
                size_t hitEntity = 0;
                glm::vec3 hitPointWorld(0.0f);
                glm::vec3 hitNormalWorld(0.0f);
                size_t emissiveEntityID = 0;
                float betaContribution = 0.0f;
                // check intersection with geometry
                GPUDataOutput::QuadraticInfo quadraticInfo(0.0f);
                bool hit = geometryIntersectionQuadric(emissiveEntityID, directLightingOrigin, directLightDir,
                                                       hitEntity,
                                                       closest_t, hitPointWorld, hitNormalWorld, betaContribution,
                                                       quadraticInfo, true);

                float tGeom = hit ? glm::length(hitPointWorld - m_cameraTransform->getPosition()) : FLT_MAX;
                float tAperture = glm::length(directLightingOrigin - m_cameraTransform->getPosition());
                //float tGeom = hit ? closest_t : FLT_MAX;
                if (tAperture < tGeom) {
                    glm::vec3 cameraHitPointWorld = directLightingOrigin + directLightDir * camera_t;

                    glm::mat4 worldToCamera = glm::inverse(m_cameraTransform->getTransform());
                    glm::vec4 hitPointCam4 = worldToCamera * glm::vec4(cameraHitPointWorld, 1.0f);
                    cameraHitPointLocal = hitPointCam4 / hitPointCam4.w;
                    return true;
                }
            }
            return false;
        }

*/

        bool geometryIntersectionQuadric(
            size_t gaussianID,
            const glm::vec3& rayOrigin,
            const glm::vec3& rayDir,
            size_t& hitEntity,
            float& closest_t,
            glm::vec3& hitPointWorld,
            glm::vec3& hitNormalWorld,
            float& betaContribution,
            GPUDataOutput::QuadraticInfo& quadraticInfo,
            bool isContributionRay = false
        ) const {
            // Set up initial values.
            float tMinGlobal = std::numeric_limits<float>::max();
            bool hitFound = false;
            size_t bestQuadricIndex = 0;
            glm::vec3 bestHitPoint(0.0f), bestHitNormal(0.0f);
            float bestBeta = 0.0f;
            // Set up an iterative traversal stack.
            const int MAX_STACK_SIZE = 64;
            int stack[MAX_STACK_SIZE];
            int stackPtr = 0;
            // Push the BVH root index (assumed 0) onto the stack.
            stack[stackPtr++] = m_gpuData.numBVHNodes - 1;

            // Traverse the BVH iteratively.
            while (stackPtr > 0) {
                int currentIndex = stack[--stackPtr];
                const BVHNode2& node = m_gpuData.bvhNodes[currentIndex];

                // Test ray against node's bounding box.
                if (!rayAABBIntersect(rayOrigin, rayDir, node.bboxMin, node.bboxMax, tMinGlobal))
                    continue;

                if (node.isLeaf) {
                    // Leaf node: perform the detailed quadric intersection test.
                    float tCandidate = std::numeric_limits<float>::max();
                    glm::vec3 localHitPoint(0.0f), localHitNormal(0.0f);
                    float beta = 0.0f;
                    const QuadricInputAssembly& quadric = m_gpuData.quadricInputAssembly[node.quadricIndex];
                    if (isContributionRay) {
                        if (checkContributionCollision(rayOrigin, rayDir, quadric, localHitPoint)) {
                            hitFound = true;
                            bestHitPoint = localHitPoint;
                        }
                    }
                    else {
                        if (intersectQuadricLeaf(rayOrigin, rayDir, quadric, tCandidate, localHitPoint, localHitNormal,
                                                 beta, quadraticInfo)) {
                            if (tCandidate < tMinGlobal) {
                                tMinGlobal = tCandidate;
                                bestQuadricIndex = node.quadricIndex;
                                bestHitPoint = localHitPoint;
                                bestHitNormal = localHitNormal;
                                bestBeta = beta;
                                hitFound = true;
                            }
                        }
                    }
                }
                else {
                    // Internal node: push its child nodes onto the stack.
                    if (stackPtr + 2 < MAX_STACK_SIZE) {
                        stack[stackPtr++] = node.leftChild;
                        stack[stackPtr++] = node.rightChild;
                    }
                }
            }

            // If a hit was found, update the output parameters.
            if (hitFound) {
                hitEntity = bestQuadricIndex;
                closest_t = tMinGlobal;
                hitPointWorld = bestHitPoint;
                hitNormalWorld = bestHitNormal;
                betaContribution = bestBeta;
                return true;
            }
            return false;
        }

        /*
        // ---------------------------------------------------------
        // Single Photon Trace (Multi-Bounce)
        // ---------------------------------------------------------
        void traceOnePhotonSingleBounceEmissiveGradient(size_t photonID) const {
            auto camera2World = m_cameraTransform->getTransform();
            glm::mat4 world2Camera = glm::inverse(camera2World);

            glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 pinholePosition = m_cameraTransform->getPosition();
            glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
            // A point on the plane

            // For clarity, rename e_o = emissionOrigin, e_d = apertureSampleDir
            glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
            glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;
            glm::vec3 a = m_gpuDataOutput[photonID].apertureHitPoint;
            //glm::vec3 hitCam = m_gpuDataOutput[photonID].cameraHitPointLocal;
            float etmin = m_gpuDataOutput[photonID].emissionDirectionLength;
            size_t gaussianID = m_gpuDataOutput[photonID].gaussianID;
            glm::vec3 a_c = m_cameraTransform->getPosition(); // center of aperture
            glm::vec3 gaussianPosition = m_gpuData.gaussianInputAssembly[gaussianID].position;
            float emissionPower = m_gpuData.gaussianInputAssembly[gaussianID].emission;
            glm::vec3 &e_c = gaussianPosition;
            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;


            glm::vec3 f = cameraPlanePointWorld; // e.g., defined in your camera parameters
            glm::vec3 f_n = cameraNormal; // e.g., (0,0,1) if the focal plane faces +Z

            // PIXEL LOSS GROUND TRUTH
            GPUDataOutput::Bounce &object = m_gpuDataOutput[photonID].bounce[0];
            size_t hitObjectID = object.quadricID;

            if (hitObjectID > m_gpuData.numGaussians || gaussianID > m_gpuData.numGaussians || !object.hitCamera) {
                return;
            }

            auto &newHitObject = m_gpuData.gaussianInputAssembly[hitObjectID];
            glm::vec3 g_c = newHitObject.position;
            glm::vec3 g_n = newHitObject.normal;
            glm::vec3 g_hit2 = object.hitPointWorld;
            float t_g = glm::dot((g_c - e_o), g_n) / glm::dot(e_d, g_n);
            glm::vec3 g_hit = e_o + t_g * e_d;
            float tg_gt = glm::dot((g_c - e_c), g_n) / glm::dot(e_d, g_n);
            glm::vec3 g_hit_gt = e_c + tg_gt * e_d;

            glm::vec3 apertureHitPoint(0.0f);
            glm::vec3 a_d_gt = sampleDirectionTowardAperture(
                g_hit_gt,
                a_c,
                cameraNormal,
                apertureHitPoint,
                0,
                photonID
            );

            glm::vec3 camHit(0.0f);
            float a_tmin_gt = 0.0f;
            float incidentAngle = 0.0f;
            bool cameraHit = checkCameraPlaneIntersection(g_hit_gt, a_d_gt, camHit,
                                                          a_tmin_gt, incidentAngle);
            glm::vec3 cameraHitPointWorldGT = g_hit_gt + a_d_gt * a_tmin_gt;

            glm::vec3 a_d = glm::normalize(a_c - g_hit);
            float a_tmin = glm::dot((f - g_hit), f_n) / (glm::dot(a_d, f_n));
            glm::vec3 cameraHitPointWorld = g_hit + a_d * a_tmin;
            glm::vec4 hitPointCam = world2Camera * glm::vec4(cameraHitPointWorld, 1.0f);
            hitPointCam = hitPointCam / hitPointCam.w;
            float px = hitPointCam.x;
            float py = hitPointCam.y;
            float pz = hitPointCam.z;
            float xPixel = (fx * px / pz) + cx;
            float yPixel = (fy * py / pz) + cy;

            if (xPixel > m_camera->m_parameters.width || yPixel > m_camera->m_parameters.height || xPixel < 0.0f ||
                yPixel < 0.0f) {
                return;
            }

            glm::vec4 hitPointCamGT = world2Camera * glm::vec4(cameraHitPointWorldGT, 1.0f);
            hitPointCamGT = hitPointCamGT / hitPointCamGT.w;
            float px_gt = hitPointCamGT.x;
            float py_gt = hitPointCamGT.y;
            float pz_gt = hitPointCamGT.z;
            float gtPixelU = (fx * px_gt / pz_gt) + cx;
            float gtPixelV = (fy * py_gt / pz_gt) + cy;

            /*
            float dLoss = bilinearSample(m_gpuData.gradientImage,
                                         static_cast<int>(m_camera->parameters().width),
                                         static_cast<int>(m_camera->parameters().height),
                                         xPixel, yPixel);

            if (dLoss == 0.0f)
                return;

            /// Finding gradient of pixel projection to e0

            glm::vec3 grad_tg = -g_n / (glm::dot(e_d, g_n));

            glm::mat3 J_ghit_eo = glm::mat3(1.0f);
            J_ghit_eo += glm::outerProduct(e_d, grad_tg);

            // d_atmin_eo
            glm::vec3 w = a_c - g_hit;
            float w_len = glm::length(w);
            glm::mat3 J_w_eo = -J_ghit_eo;

            glm::vec3 tmp = w / static_cast<float>(std::pow(w_len, 3));
            glm::vec3 tmp2 = glm::transpose(J_w_eo) * w;
            glm::mat3 term2 = glm::outerProduct(tmp, tmp2);

            glm::mat3 J_ad_eo = (-1 / w_len) * J_w_eo;
            J_ad_eo += term2;

            // grad atmin_eo
            float d = glm::dot(a_d, f_n);
            glm::vec3 d_d = f_n * J_ad_eo;

            float n = glm::dot((f - g_hit), f_n);
            glm::vec d_n = -f_n * J_ghit_eo;

            glm::vec3 grad_atmin_eo = ((d_n * d) - (n * d_d)) / (d * d);

            // J_p_eo

            glm::mat3 J_p_eo = J_ghit_eo;
            J_p_eo += glm::outerProduct(a_d, grad_atmin_eo) + a_tmin * J_ad_eo;


            float px_camera = hitPointCam.x;
            float py_camera = hitPointCam.y;
            float pz_camera = hitPointCam.z;
            // Compute derivatives
            float inv_pz = 1.0f / pz_camera;
            float inv_pz2 = inv_pz * inv_pz; // 1/pz^2
            // Construct Jacobian matrix J_(u,v),p (2x3)
            glm::mat3x3 J_uv_p(0.0f); // J_uv_p is in fact a 2x3 matrix but use a 3x3 for simple integration with glm
            J_uv_p[0][0] = fx * inv_pz; // ∂u/∂px
            J_uv_p[0][1] = 0.0f; // ∂u/∂py
            J_uv_p[0][2] = -fx * px_camera * inv_pz2; // ∂u/∂pz

            J_uv_p[1][0] = 0.0f; // ∂v/∂px
            J_uv_p[1][1] = fy * inv_pz; // ∂v/∂py
            J_uv_p[1][2] = -fy * py_camera * inv_pz2; // ∂v/∂pz
            //glm::mat2x3  J_uv_eo = multiply2x3_3x3(J_uv_p, dp_de_o_camera);
            // Apply Rotation:
            glm::mat3 w2c = glm::mat3(world2Camera);
            glm::mat3 J_p_eo_camera = w2c * J_p_eo;
            glm::mat3 J_uv_eo = glm::transpose(J_uv_p) * J_p_eo_camera;

            // 1) Evaluate the pixel mismatch:
            float du = (xPixel - gtPixelU);
            float dv = (yPixel - gtPixelV);

            // 2) dL/d(u) and dL/d(v) for L2 cost:
            float dLdu = 2.f * du;
            float dLdv = 2.f * dv;

            // For the geometry part, you need to pull back the loss derivative in image space through the Jacobian:
            glm::vec3 grad_geometry(0.0f);
            grad_geometry.x = (dLdu * J_uv_eo[0][0] + dLdv * J_uv_eo[0][1]);
            grad_geometry.y = (dLdu * J_uv_eo[1][0] + dLdv * J_uv_eo[1][1]);
            grad_geometry.z = (dLdu * J_uv_eo[2][0] + dLdv * J_uv_eo[2][1]);

            //glm::vec3 total_gradient = grad_geometry * dLoss;

            // Atomically accum ulate the gradient.
            sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    sum_x(m_gpuData.gaussianGradients[gaussianID].x),
                    sum_y(m_gpuData.gaussianGradients[gaussianID].y),
                    sum_z(m_gpuData.gaussianGradients[gaussianID].z);

            //sum_x.fetch_add(total_gradient.x);
            //sum_y.fetch_add(total_gradient.y);
            //sum_z.fetch_add(total_gradient.z);
        }

*/
    };
}

#endif //PATHTRACER2DGSKERNELBACKWARD_H
