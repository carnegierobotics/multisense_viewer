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
                                  GPUDataOutput *gpuDataOutput,
                                  PCG32 *rng)
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
        GPUDataOutput *m_gpuDataOutput{};

        PCG32 *m_rng;
        TransformComponent *m_cameraTransform{};
        PinholeCamera *m_camera{};

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

        // ---------------------------------------------------------
        // Single Photon Trace (Single-Bounce)
        // ---------------------------------------------------------
        void traceOnePhotonSingleBounceObjectGradient(size_t photonID) const {
            GPUDataOutput::Bounce &object = m_gpuDataOutput[photonID].bounce[0];
            size_t hitObjectID = object.quadricID;
            if (hitObjectID > m_gpuData.numQuadrics) {
                return;
            }

            auto camera2World = m_cameraTransform->getTransform();
            glm::mat4 world2Camera = glm::inverse(camera2World);

            glm::vec3 cameraNormal = glm::normalize(glm::mat3(camera2World) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 pinholePosition = m_cameraTransform->getPosition();
            glm::vec3 cameraPlanePointWorld = glm::vec3(camera2World * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));
            glm::vec3 a_c = m_cameraTransform->getPosition(); // center of aperture


            size_t gaussianID = m_gpuDataOutput[photonID].gaussianID;
            glm::vec3 e_c = m_gpuData.gaussianInputAssembly[gaussianID].position;


            glm::vec3 f = cameraPlanePointWorld; // e.g., defined in your camera parameters
            glm::vec3 f_n = cameraNormal; // e.g., (0,0,1) if the focal plane faces +Z
            if (!object.hitCamera) {
                return;
            }


            /*
            glm::vec3 a_d = glm::normalize(a_c - g_hit);
            float a_tmin = glm::dot((f - g_hit), f_n) / (glm::dot(a_d, f_n));
            glm::vec3 cameraHitPointWorld = g_hit + a_d * a_tmin;
            glm::vec4 hitPointCam = world2Camera * glm::vec4(cameraHitPointWorld, 1.0f);
            hitPointCam = hitPointCam / hitPointCam.w;
            */

            auto &quadric = m_gpuData.quadricInputAssembly[hitObjectID];

            /*
            float closest_t = FLT_MAX;
            size_t hitEntity = 0;
            glm::vec3 hitPointWorld(0.0f);
            glm::vec3 hitNormalWorld(0.0f);

            glm::vec3 delta = a_c - quadric.transform.getPosition();
            glm::vec3 ad_qc = glm::normalize(delta);
            float betaContribution = 0.0f;
            GPUDataOutput::QuadraticInfo info;
            // check intersection with geometry
            bool hit = geometryIntersectionQuadric(gaussianID, quadric.transform.getPosition(), ad_qc, hitEntity,
                                                   closest_t,
                                                   hitPointWorld,
                                                   hitNormalWorld, betaContribution, info, true);
            if (hit)
                return;
            */


            float xPixel = object.pixelCoordinate.x;
            float yPixel = object.pixelCoordinate.y;

            int x0 = static_cast<int>(std::floor(xPixel));
            int y0 = static_cast<int>(std::floor(yPixel));
            float dx = xPixel - x0;
            float dy = yPixel - y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // 3. Compute the bilinear weights for the 4 pixels.
            float w00 = (1.0f - dx) * (1.0f - dy); // weight for pixel at (x0, y0)
            float w10 = dx * (1.0f - dy); // weight for pixel at (x1, y0)
            float w01 = (1.0f - dx) * dy; // weight for pixel at (x0, y1)
            float w11 = dx * dy; // weight for pixel at (x1, y1)

            // 5. Retrieve image dimensions.
            const size_t imageWidth = m_camera->parameters().width;
            const size_t imageHeight = m_camera->parameters().height;

            // Helper lambda to add the weighted flux contribution to a given pixel.
            auto addFluxToPixel = [&](int px, int py, float weight) {
                // Only update if the pixel is inside the image bounds.
                if (px >= 0 && px < static_cast<int>(imageWidth) &&
                    py >= 0 && py < static_cast<int>(imageHeight)) {
                    size_t pixelIndex = static_cast<size_t>(py) * imageWidth + static_cast<size_t>(px);
                    m_gpuData.gradientImagePerObject[pixelIndex] = static_cast<float>(hitObjectID);
                }
            };
            // 6. Distribute the corrected flux into the four neighboring pixels.
            addFluxToPixel(x0, y0, w00);
            addFluxToPixel(x1, y0, w10);
            addFluxToPixel(x0, y1, w01);
            addFluxToPixel(x1, y1, w11);


            ////////// GT CALCULATION /&///////
            ///            glm::vec3 q_c = quadric.transform.getPosition();

            // If quadric.transform is orthonormal, its inverse is its transpose.
            /*
            glm::vec3 a_d_gt = glm::normalize(a_c - q_c);
            float a_tmin_gt = glm::dot((f - q_c), f_n) / (glm::dot(a_d_gt, f_n));
            glm::vec3 cameraHitPointWorld_gt = q_c + a_d_gt * a_tmin_gt;
            glm::vec4 hitPointCam_gt = world2Camera * glm::vec4(cameraHitPointWorld_gt, 1.0f);
            hitPointCam_gt = hitPointCam_gt / hitPointCam_gt.w;
            float px_gt = hitPointCam_gt.x;
            float py_gt = hitPointCam_gt.y;
            float pz_gt = hitPointCam_gt.z;

*/
            glm::mat3 w2c = glm::mat3(world2Camera);
            glm::mat3 I = glm::mat3(1.0f);

            float alpha_x = tanhf(quadric.t_x);
            float alpha_y = tanhf(quadric.t_y);
            auto quadric2World = glm::mat3(quadric.transform.getTransform()); // mat4
            glm::mat3 world2Quadric = glm::transpose(glm::mat3(quadric2World));

            auto world2QuadricTransform = glm::inverse(quadric.transform.getTransform());
            // ===== Backward Pass =====

            glm::vec3 q_hit_world = object.hitPointWorld;
            float px = object.cameraHitPointLocal.x;
            float py = object.cameraHitPointLocal.y;
            float pz = object.cameraHitPointLocal.z;
            glm::vec3 a_d = object.apertureDirection;
            float g_tmin = object.emissionDirectionLength;
            float a_tmin = object.cameraDirectionLength;


            glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
            glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;

            glm::vec3 e_d_local = object.quadInfo.localRayDirection;
            glm::vec3 e_o_local = object.quadInfo.localRayOrigin;

            auto &quadInfo = object.quadInfo;
            float A = quadInfo.A;
            float B = quadInfo.B;
            float discriminant = quadInfo.discriminant;
            // Transform the local hit point back to world space.
            if (abs(A) < std::numeric_limits<float>::epsilon() || discriminant < std::numeric_limits<float>::epsilon()
                || quadInfo.rootIndex == -1) {
                return;
                }


            // We now compute the gradient (jacobian) of our hit point and subsequent losses with respect to q_c.
            // --- (1) Gradients of B and C with respect to q_c ---
            glm::vec3 grad_B_eo;
            grad_B_eo.x = quadric.c * (2.0f * alpha_x * e_d_local.x) / (quadric.a * quadric.a);
            grad_B_eo.y = quadric.c * (2.0f * alpha_y * e_d_local.y) / (quadric.b * quadric.b);
            grad_B_eo.z = 0.0f; // B does not depend on e_o_local.z

            // Thus, derivative of B with respect to q_c:
            glm::vec3 dB_dqc = -glm::transpose(world2Quadric) * grad_B_eo;

            glm::vec3 grad_C_eo;
            grad_C_eo.x = quadric.c * 2.0f * alpha_x * e_o_local.x / (quadric.a * quadric.a);
            grad_C_eo.y = quadric.c * 2.0f * alpha_y * e_o_local.y / (quadric.b * quadric.b);
            grad_C_eo.z = -1;
            glm::vec3 dC_dqc = -glm::transpose(world2Quadric) * grad_C_eo;

            // --- (2) Derivatives of t_min with respect to B and C ---
            float sqrtDiscriminant = std::sqrt(discriminant);
            float inv2A = 1.0f / (2.0f * A);
            float BoverDisc = B / sqrtDiscriminant;

            float d_tmin_dC = 0.0f;
            float d_tmin_dB = 0.0f;

            if (quadInfo.rootIndex == 1) {
                // minus‑root
                d_tmin_dB = -inv2A * (1.0f + BoverDisc);
                d_tmin_dC = 1.0f / sqrtDiscriminant;
            } else if (quadInfo.rootIndex == 2) {
                // plus‑root
                d_tmin_dB = inv2A * (-1.0f + BoverDisc);
                d_tmin_dC = -1.0f / sqrtDiscriminant;
            } else {
                return;
            }

            // --- (3) Chain rule: derivative of t_min with respect to q_c ---
            glm::vec3 dtmin_dqc = d_tmin_dB * dB_dqc + d_tmin_dC * dC_dqc;

            // --- (4) Derivative of q_hit_local ---
            glm::mat3 d_hitLocal_dqc = -world2Quadric + glm::outerProduct(e_d_local, dtmin_dqc);

            // --- (5) Derivative of q_hit with respect to q_c ---
            // Recall: q_hit = quadric2World * q_hit_local + q_c, so:
            glm::mat3 d_ghit_dqc = quadric2World * d_hitLocal_dqc + glm::mat3(1.0f);
            // d_ghit_dqc is our J_ghit,qc.

            // --- (6) Derivative of a_d = normalize(a_c - q_hit) ---
            glm::vec3 v_tmp = a_c - q_hit_world;
            float v_len = glm::length(v_tmp);
            // The derivative of a normalized vector: (I/v_len - outer(v_tmp,v_tmp)/(v_len³))
            glm::mat3 J_ad_qc = (I / v_len - (glm::outerProduct(v_tmp, v_tmp) / (v_len * v_len * v_len))) * (-
                                    d_ghit_dqc);

            // --- (7) Derivative of a_tmin = ((f - q_hit) ⋅ f_n) / (a_d ⋅ f_n) ---
            float n_val = glm::dot(f - q_hit_world, f_n);
            float d_val = glm::dot(a_d, f_n);
            // d(n)/d(q_c) = - (transpose(J_ad_qc) * f_n)
            glm::vec3 d_n = glm::transpose(d_ghit_dqc) * (-f_n);
            // d(d)/d(q_c) = (transpose(J_ad_qc) * f_n)
            glm::vec3 d_d = glm::transpose(J_ad_qc) * f_n;
            glm::vec3 nabla_atmin_qc = (d_val * d_n - n_val * d_d) / (d_val * d_val);

            // --- (8) Derivative of the focal plane intersection p(q_c) = q_hit + a_tmin * a_d ---
            // Using the product rule:
            glm::mat3 term1 = d_ghit_dqc;
            glm::mat3 term2 = glm::transpose(glm::outerProduct(nabla_atmin_qc, a_d));
            glm::mat3 term3 = a_tmin * J_ad_qc;
            glm::mat3 J_p_qc = term1 + term2 + term3;

            //glm::mat3 J_p_qc = d_ghit_dqc + glm::outerProduct(a_d, nabla_atmin_qc) + a_tmin * J_ad_qc;
            // --- (9) Camera extrinsics: p_camera = R_w2c * p(q_c) ---

            glm::mat3 J_pc_qc = w2c * J_p_qc;
            // --- (10) Pinhole projection derivative ---
            // For a camera point p_camera = (px,py,pz), the projection is:
            // u = fx * px / pz + cx, v = fy * py / pz + cy.
            // Its Jacobian (2×3) is:
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            glm::mat3 J_uv_pcam(0.0f);
            J_uv_pcam[0][0] = fx / pz;
            J_uv_pcam[0][1] = 0.0f;
            J_uv_pcam[0][2] = -fx * px / (pz * pz);
            J_uv_pcam[1][0] = 0.0f;
            J_uv_pcam[1][1] = fy / pz;
            J_uv_pcam[1][2] = -fy * py / (pz * pz);
            J_uv_pcam = glm::transpose(J_uv_pcam);
            // --- (11) Derivative of pixel coordinates with respect to q_c ---
            glm::mat3x3 J_uv_qc = J_uv_pcam * J_pc_qc;
            glm::mat3 total_gradient = J_uv_qc; // Scale with the beta contribution for some reason
            // Atomically accum ulate the gradient.
            m_gpuData.gradientPixelCoordinates[photonID] = glm::vec2(xPixel, yPixel);
            m_gpuData.photonIDGradient[photonID] = total_gradient;

                        // Beta kernel derivative:
            float gd = quadInfo.geodesic;
            float p_tmp = 4* exp(quadric.b_beta);
            float db_dgd = -2 * p_tmp * gd * std::pow((1-(gd * gd)), p_tmp - 1);


            /*
            if (base <= 0.0f) {
                db_dgd = 0.0f;
            } else {
                float p_tmp = 4.0f * std::exp(quadric.b_beta);
                db_dgd = -2.0f * p_tmp * gd * std::pow(base, p_tmp - 1.0f);
            }
            */
            // Local hit coordinates
            float x = quadInfo.hitLocal.x;
            float y = quadInfo.hitLocal.y;
            float rho   = quadInfo.rho;
            float theta = quadInfo.theta;

            // Precompute common denominators
            float denom = x*x + y*y;
            float twoAtRho = 2.0f * quadInfo.a_theta * rho;
            float sqrtTerm = std::sqrt(1.0f + twoAtRho*twoAtRho);

            // 1) dρ/dx, dρ/dy
            float d_rho_x   = x / rho;
            float d_rho_y   = y / rho;

            // 2) dθ/dx, dθ/dy
            float d_theta_x = -y / denom;
            float d_theta_y =  x / denom;

            // 3) d(aθ)/dθ
            float a       = quadric.a;
            float b       = quadric.b;
            float c       = quadric.c;
            float d_aθ_dθ = 2.0f * c * std::cos(theta) * std::sin(theta) *
                            (alpha_y/(b*b) - alpha_x/(a*a));

            // 4) d(gd)/dρ
            float d_gd_drho = 0.5f * sqrtTerm
                            + (rho * (4.0f * quadInfo.a_theta*quadInfo.a_theta * rho)) / (4.0f * sqrtTerm)
                            + 0.5f / sqrtTerm;

            // 5) d(gd)/d(aθ)
            float d_gd_daθ = -1.0f/(4.0f * quadInfo.a_theta * quadInfo.a_theta) * std::asinh(twoAtRho)
                            + (0.5f * rho) / (quadInfo.a_theta * sqrtTerm);

            // 6) d(gd)/dx, d(gd)/dy
            float d_gd_x = d_theta_x * d_gd_daθ * d_aθ_dθ + d_rho_x * d_gd_drho;
            float d_gd_y = d_theta_y * d_gd_daθ * d_aθ_dθ + d_rho_y * d_gd_drho;

            // 7) dβ/dx, dβ/dy
            float d_beta_x = db_dgd * d_gd_x;
            float d_beta_y = db_dgd * d_gd_y;

            // Final 2‑D gradient in local quadric coordinates
            glm::vec2 nabla_beta_xy(d_beta_x, d_beta_y);

            // Precompute dZ/dx, dZ/dy on the local surface
            float dZdx = 2.0f * quadric.c * alpha_x * quadInfo.hitLocal.x / (quadric.a * quadric.a);
            float dZdy = 2.0f * quadric.c * alpha_y * quadInfo.hitLocal.y / (quadric.b * quadric.b);

            // 1) ∂P/∂x and ∂P/∂y in world space
            glm::vec3 dPdx_world = quadric2World * glm::vec3(1.0f, 0.0f, dZdx);
            glm::vec3 dPdy_world = quadric2World * glm::vec3(0.0f, 1.0f, dZdy);

            // 2) Transform into camera space (treating these as direction vectors → w=0)
            glm::vec3 dPdx_cam = world2Camera * glm::vec4(dPdx_world, 0.0f);
            glm::vec3 dPdy_cam = world2Camera * glm::vec4(dPdy_world, 0.0f);

            // 3) Project into pixel space via your 2×3 J_uv_p_camera
            glm::vec2 duv_dx = J_uv_pcam * dPdx_cam;
            glm::vec2 duv_dy = J_uv_pcam * dPdy_cam;

            // Assemble the 2×2 Jacobian J_uv_xy
            glm::mat2 J_uv_xy(
                duv_dx.x, duv_dy.x,
                duv_dx.y, duv_dy.y
            );

            // 4) Invert & multiply by -∇_{xy}β
            glm::mat2 invJ = glm::inverse(J_uv_xy);
            glm::vec2 nabla_uv_beta = invJ * nabla_beta_xy;


            int xPixelInt = std::round(xPixel);
            int yPixelInt = std::round(yPixel);
            size_t pixelIndex = yPixelInt * imageWidth + xPixelInt;
            m_gpuData.gradientImageU[pixelIndex] = nabla_uv_beta.x;
            m_gpuData.gradientImageV[pixelIndex] = nabla_uv_beta.y;

            if (nabla_uv_beta.x < -10.0f || nabla_uv_beta.y < -10.0f) {
                int debug = 1;
            }
            switch (hitObjectID) {
                case 0: {
                    int debug = 1;
                }
                break;
                case 1: {
                    int debug = 1;
                }
                break;
                case 2: {
                    int debug = 1;
                }
                break;
                case 3: {
                    int debug = 1;
                }
                break;
                default: {
                    int debug = 1;
                }
            }

            // Center Quadric gradients

            /*
            glm::mat3 J_uv_gt_pcam(0.0f);
            J_uv_gt_pcam[0][0] = fx / pz_gt;
            J_uv_gt_pcam[0][1] = 0.0f;
            J_uv_gt_pcam[0][2] = -fx * px_gt / (pz_gt * pz_gt);
            J_uv_gt_pcam[1][0] = 0.0f;
            J_uv_gt_pcam[1][1] = fy / pz_gt;
            J_uv_gt_pcam[1][2] = -fy * py_gt / (pz_gt * pz_gt);
            J_uv_gt_pcam = glm::transpose(J_uv_gt_pcam);

            // --- (6) Derivative of a_d = normalize(a_c - q_hit) ---
            glm::vec3 v_tmp_gt = a_c - q_c;
            float v_len_gt = glm::length(v_tmp_gt);
            // The derivative of a normalized vector: (I/v_len_gt - outer(v_tmp_gt,v_tmp_gt)/(v_len_gt³))
            glm::mat3 J_ad_gt_qc = (I / v_len_gt - glm::outerProduct(v_tmp_gt, v_tmp_gt) / (
                                        v_len_gt * v_len_gt * v_len_gt)) * (-I);


            // --- (7) Derivative of a_tmin = ((f - q_hit) ⋅ f_n) / (a_d ⋅ f_n) ---
            float num = glm::dot(f - q_c, f_n);
            float den = glm::dot(a_d_gt, f_n);
            // d(n)/d(q_c) = - (transpose(J_ad_qc) * f_n)
            glm::vec3 d_num = -f_n;
            // d(d)/d(q_c) = (transpose(J_ad_gt_qc) * f_n)
            glm::vec3 d_den = glm::transpose(J_ad_gt_qc) * f_n;
            glm::vec3 grad_atmin_gt_qc = (den * d_num - num * d_den) / (den * den);

            glm::mat3 term2_gt = glm::transpose(glm::outerProduct(grad_atmin_gt_qc, a_d_gt));
            glm::mat3 term3_gt = a_tmin_gt * J_ad_gt_qc;
            glm::mat3 J_p_gt_qc = I + term2_gt + term3_gt;

            glm::mat3 J_pc_gt_qc = w2c * J_p_gt_qc;

            glm::mat3 J_uv_gt_qc = J_uv_gt_pcam * J_pc_gt_qc;
            */


            /*
            glm::vec3 q_c = newHitObject.position;
            glm::vec3 g_n = newHitObject.normal;
            float t_g = glm::dot((q_c - e_o), g_n) / glm::dot(e_d, g_n);
            glm::vec3 q_hit = e_o + t_g * e_d;
            float tg_gt = glm::dot((q_c - e_c), g_n) / glm::dot(e_d, g_n);
            glm::vec3 q_hit_gt = e_c + tg_gt * e_d;

            glm::vec3 apertureHitPoint(0.0f);
            glm::vec3 a_d_gt = sampleDirectionTowardAperture(
                q_hit_gt,
                a_c,
                cameraNormal,
                apertureHitPoint,
                0,
                photonID
            );

            glm::vec3 camHit(0.0f);
            float a_tmin_gt = 0.0f;
            float incidentAngle = 0.0f;
            bool cameraHit = checkCameraPlaneIntersection(q_hit_gt, a_d_gt, camHit,
                                                          a_tmin_gt, incidentAngle);
            glm::vec3 cameraHitPointWorldGT = q_hit_gt + a_d_gt * a_tmin_gt;

            glm::vec3 a_d = glm::normalize(a_c - q_hit);
            float a_tmin = glm::dot((f - q_hit), f_n) / (glm::dot(a_d, f_n));
            glm::vec3 cameraHitPointWorld = q_hit + a_d * a_tmin;
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




            // Intersection derivative from light source
            glm::mat3 J_ghit_qc = (1 / glm::dot(e_d, g_n)) * glm::outerProduct(e_d, g_n);

            // outgoing direction derivative to camera
            glm::vec3 v_tmp = a_c - q_hit;
            float v_len = glm::length(v_tmp);

            //
            // Build the bracket [ I/|v| - (v v^T)/|v|^3 ]:
            glm::mat3 bracket = glm::mat3(1.0f) * (1.0f / v_len);
            bracket -= (glm::outerProduct(v_tmp, v_tmp) / (v_len * v_len * v_len));

            // Then multiply by -J_ghit_qc:
            glm::mat3 J_ad_qc = bracket * (-J_ghit_qc);

            // intersection parameter to camera
            float n = glm::dot((f - a_d), f_n);
            glm::vec3 d_n = (-f_n) * J_ad_qc;

            float d = glm::dot(a_d, f_n);
            glm::vec3 d_d = f_n * J_ad_qc;

            glm::vec3 tmp_numerator = (d * d_n) - (n * d_d);
            float tmp_denom = d * d;

            glm::vec3 grad_atmin_qc = tmp_numerator / tmp_denom;

            // Focal Plane intersection coordinates:
            glm::mat3 tmp_term_2 = glm::outerProduct(a_d, grad_atmin_qc);
            glm::mat3 tmp_term_3 = a_tmin * J_ad_qc;
            glm::mat3 J_p_qc = J_ghit_qc + tmp_term_2 + tmp_term_3;

            // Camera extrinsics gradients:

            glm::mat3 w2c = glm::mat3(world2Camera);
            glm::mat3 J_pc_qc = w2c * J_p_qc;

            // Construct Jacobian matrix J_(u,v),p (2x3)
            float px_camera = hitPointCam.x;
            float py_camera = hitPointCam.y;
            float pz_camera = hitPointCam.z;
            // Compute derivatives
            float inv_pz = 1.0f / pz_camera;
            float inv_pz2 = inv_pz * inv_pz; // 1/pz^2
            glm::mat3x3 J_uv_p(0.0f); // J_uv_p is in fact a 2x3 matrix but use a 3x3 for simple integration with glm
            J_uv_p[0][0] = fx * inv_pz; // ∂u/∂px
            J_uv_p[0][1] = 0.0f; // ∂u/∂py
            J_uv_p[0][2] = -fx * px_camera * inv_pz2; // ∂u/∂pz

            J_uv_p[1][0] = 0.0f; // ∂v/∂px
            J_uv_p[1][1] = fy * inv_pz; // ∂v/∂py
            J_uv_p[1][2] = -fy * py_camera * inv_pz2; // ∂v/∂pz

            glm::mat3 J_uv_eo = glm::transpose(J_uv_p) * J_pc_qc;

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

            */
        }

        bool geometryIntersectionQuadric(
            size_t gaussianID,
            const glm::vec3 &rayOrigin,
            const glm::vec3 &rayDir,
            size_t &hitEntity,
            float &closest_t,
            glm::vec3 &hitPointWorld,
            glm::vec3 &hitNormalWorld,
            float &betaContribution,
            GPUDataOutput::QuadraticInfo &quadraticInfo,
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
                const BVHNode &node = m_gpuData.bvhNodes[currentIndex];

                // Test ray against node's bounding box.
                if (!rayAABBIntersect(rayOrigin, rayDir, node.bboxMin, node.bboxMax, tMinGlobal))
                    continue;

                if (node.isLeaf) {
                    // Leaf node: perform the detailed quadric intersection test.
                    float tCandidate = std::numeric_limits<float>::max();
                    glm::vec3 localHitPoint(0.0f), localHitNormal(0.0f);
                    float beta = 0.0f;
                    const QuadricInputAssembly &quadric = m_gpuData.quadricInputAssembly[node.quadricIndex];
                    if (isContributionRay) {
                        if (checkContributionCollision(rayOrigin, rayDir, quadric, localHitPoint)) {
                            hitFound = true;
                            bestHitPoint = localHitPoint;
                        }
                    } else {
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
                } else {
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
                */
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


        // ---------------------------------------------------------
        // Single Photon Trace
        // ---------------------------------------------------------
        void traceOnePhotonDirectLighting(size_t photonID) const {
            if (!m_gpuDataOutput[photonID].hitCamera)
                return;

            /*

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
            glm::vec3 hitCam = m_gpuDataOutput[photonID].cameraHitPointLocal;
            float etmin = m_gpuDataOutput[photonID].emissionDirectionLength;
            size_t gaussianID = m_gpuDataOutput[photonID].gaussianID;

            glm::vec3 gaussianPosition = m_gpuData.gaussianInputAssembly[gaussianID].position;
            float emissionPower = m_gpuData.gaussianInputAssembly[gaussianID].emission;

            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float px = hitCam.x;
            float py = hitCam.y;
            float pz = hitCam.z;
            float xPixel = (fx * px / pz) + cx;
            float yPixel = (fy * py / pz) + cy;

            float dLoss = bilinearSample(m_gpuData.gradientImage,
                                         static_cast<int>(m_camera->parameters().width),
                                         static_cast<int>(m_camera->parameters().height),
                                         xPixel, yPixel);

            glm::vec3 f = cameraPlanePointWorld; // e.g., defined in your camera parameters
            glm::vec3 f_n = cameraNormal; // e.g., (0,0,1) if the focal plane faces +Z

            // PIXEL LOSS GROUND TRUTH
            glm::vec3 apertureHitPoint;
            glm::vec3 e_c_gt = gaussianPosition;
            glm::vec3 e_d_gt = sampleDirectionTowardAperture(
                e_c_gt,
                m_cameraTransform->getPosition(), // center of aperture
                cameraNormal,
                apertureHitPoint,
                0,
                photonID
            );
            // Check if contribution ray intersects geometry
            // Create contribution Rays and trace towards the camera
            // Trace our contribution ray
            glm::vec3 camHit;
            float tCam;
            float incidentAngle;
            bool cameraHit = checkCameraPlaneIntersection(e_c_gt, e_d_gt, camHit,
                                                          tCam, incidentAngle);
            glm::vec3 cameraHitPointWorld = e_c_gt + e_d_gt * tCam;

            // 1. Transform the hit point from world space to camera space
            glm::vec4 hitPointCam = world2Camera * glm::vec4(cameraHitPointWorld, 1.0f);
            hitPointCam = hitPointCam / hitPointCam.w;
            float px_gt = hitPointCam.x;
            float py_gt = hitPointCam.y;
            float pz_gt = hitPointCam.z;
            float gtPixelU = (fx * px_gt / pz_gt) + cx;
            float gtPixelV = (fy * py_gt / pz_gt) + cy;

            /// PART ONE ///
            // Get the Gaussian emitter parameters.
            float sigma = m_gpuData.gaussianInputAssembly[gaussianID].scale.x; // assume uniform sigma
            glm::vec3 e_c = gaussianPosition; // center of the Gaussian emitter (optimized parameter)

            // Compute the Gaussian intensity.
            glm::vec3 delta = e_c - e_o;
            float delta_norm_sq = glm::dot(delta, delta);
            float e_i = emissionPower * std::exp(-delta_norm_sq / (2.0f * sigma * sigma));
            // Compute derivative of the Gaussian intensity with respect to the sampled position e_o.
            // Note: de_i/de_o = e_i * (e_c - e_o) / sigma^2.
            glm::vec3 de_i_deo = (e_i / (sigma * sigma)) * delta;

            /// PART TWO ///
            // Example: add the emission to that pixel
            // ...
            // Now do the backward pass to accumulate ∂L/∂e_o:
            // 1) Get dLoss/dI[u,v] from your gradient buffer

            // 2) We need the local partial derivative: (u,v) w.r.t. e_o
            //    We'll replicate the chain rule steps with your known transformations.
            //    The code below is just a skeleton; fill in details carefully.
            // 2.1) Emission Direction Derivative:
            // (a) r = a - e_o, e_d = r / norm(r)
            //     J_{e_d,e_o} = ...
            glm::vec3 r = a - e_o;
            float r_length = glm::length(r);
            glm::mat3x3 Jed_eo = (glm::outerProduct(r, r) / (r_length * r_length * r_length)) - (1 / r_length) *
                                 glm::mat3(1.0f);
            // 2.2) Focal Plane intersection parameter:
            // (b) tMin = ...
            //     dtMin/de_o = ...
            // Dot product for denominator
            float denom = glm::dot(e_d, f_n); // Scalar
            float denom_squared = denom * denom; // Avoid recomputing later
            glm::vec3 term1 = -f_n * denom; // Scalar * Vector = Vector
            glm::vec3 term2 = (glm::dot(f, f_n) - glm::dot(e_o, f_n)) * (Jed_eo * f_n);
            glm::vec3 etmin_de_o = (term1 - term2) / denom_squared; // Element-wise division
            // 2.3) Derivatives for intersections with the focal plane
            // (c) p = e_o + tMin * e_d
            //     dp/de_o = ...
            // Identity matrix (3x3)
            glm::mat3 I(1.0f);
            // Compute first term: I
            glm::mat3 dp_de_o = I;
            // Compute second term: (∂e_tmin / ∂e_o) * e_d (3x3 * 3x1 = 3x3)
            dp_de_o += glm::outerProduct(e_d, etmin_de_o);
            // Compute third term: e_tmin * (∂e_d / ∂e_o)  (scalar * 3x3 = 3x3)
            dp_de_o += etmin * Jed_eo;
            // 2.4) Derivatives for the pinhole projection
            // (d) project p -> (u,v).  Then chain:
            //     d(u,v)/de_o = J_{(u,v),p} * dp/de_o
            float px_camera = hitCam.x;
            float py_camera = hitCam.y;
            float pz_camera = hitCam.z;
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

            // build the relevant Jacobians:
            // 5) chain them: J_uv_eo = J_uv_p * dp_de_o => (2×3) * (3×3) = 2×3
            // Apply Rotation:
            glm::mat3 dp_de_o_world = glm::mat3(world2Camera) * dp_de_o;
            //glm::mat2x3  J_uv_eo = multiply2x3_3x3(J_uv_p, dp_de_o_camera);
            glm::mat3 J_uv_eo = J_uv_p * dp_de_o_world;
            // Now, combine the derivative contributions.
            // Previously, we computed dL/de_o = dLoss * de_i/de_o.
            // We add the extra term from the path-length differentiation:
            glm::vec3 grad_intensity = dLoss * de_i_deo;

            // 1) Evaluate the pixel mismatch:
            float du = (xPixel - gtPixelU);
            float dv = (yPixel - gtPixelV);

            // 2) dL/d(u) and dL/d(v) for L2 cost:
            float dLdu = 2.f * du;
            float dLdv = 2.f * dv;

            // For the geometry part, you need to pull back the loss derivative in image space through the Jacobian:
            glm::vec3 grad_geometry;
            grad_geometry.x = (dLdu * J_uv_eo[0][0] + dLdv * J_uv_eo[0][1]);
            grad_geometry.y = (dLdu * J_uv_eo[1][0] + dLdv * J_uv_eo[1][1]);
            grad_geometry.z = (dLdu * J_uv_eo[2][0] + dLdv * J_uv_eo[2][1]);

            //grad_geometry = grad_geometry * M_cam2world;
            float scaleFactor = 20000.0f;
            glm::vec3 grad_geometry_scaled = (grad_geometry * dLoss);
            // The total gradient is the sum:
            glm::vec3 grad_total = grad_geometry_scaled;

            float d_uv_x = grad_geometry_scaled.x;
            float d_uv_y = grad_geometry_scaled.y;
            float d_uv_z = grad_geometry_scaled.z;

            float d_ei_x = grad_intensity.x;
            float d_ei_y = grad_intensity.y;
            float d_ei_z = grad_intensity.z;

            float d_total_x = grad_total.x;
            float d_total_y = grad_total.y;
            float d_total_z = grad_total.z;

            // Atomically accumulate the gradient.
            sycl::atomic_ref<float, sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    sum_x(m_gpuData.gaussianGradients[gaussianID].x),
                    sum_y(m_gpuData.gaussianGradients[gaussianID].y),
                    sum_z(m_gpuData.gaussianGradients[gaussianID].z);

            if (photonID == 10000) {
                int stop = 1;
            }

            sum_x.fetch_add(grad_total.x);
            sum_y.fetch_add(grad_total.y);
            sum_z.fetch_add(grad_total.z);
            */

        }

        /*
        // Example: add the emission to that pixel
        // ...
        // Now do the backward pass to accumulate ∂L/∂e_o:
        // 1) Get dLoss/dI[u,v] from your gradient buffer

        // 2) We need the local partial derivative: (u,v) w.r.t. e_o
        //    We'll replicate the chain rule steps with your known transformations.
        //    The code below is just a skeleton; fill in details carefully.
        // 2.1) Emission Direction Derivative:
        // (a) r = a - e_o, e_d = r / norm(r)
        //     J_{e_d,e_o} = ...
        glm::vec3 r = a - e_o;
        float r_length = glm::length(r);
        glm::mat3x3 Jed_eo = (glm::outerProduct(r, r) / (r_length * r_length * r_length)) - (1 / r_length) *
            glm::mat3(1.0f);
        // 2.2) Focal Plane intersection parameter:
        // (b) tMin = ...
        //     dtMin/de_o = ...
        glm::vec3 f_n = cameraNormal;
        glm::vec3 f = cameraPlanePointWorld;
        // Dot product for denominator
        float denom = glm::dot(e_d, f_n); // Scalar
        float denom_squared = denom * denom; // Avoid recomputing later
        glm::vec3 term1 = -f_n * denom; // Scalar * Vector = Vector
        glm::vec3 term2 = (glm::dot(f, f_n) - glm::dot(e_o, f_n)) * (Jed_eo * f_n);
        // Scalar * (Matrix * Vector) = Vector
        glm::vec3 etmin_de_o = (term1 - term2) / denom_squared; // Element-wise division

        // 2.3) Derivatives for intersections with the focal plane
        // (c) p = e_o + tMin * e_d
        //     dp/de_o = ...
        // Identity matrix (3x3)
        glm::mat3 I(1.0f);
        // Compute first term: I
        glm::mat3 dp_de_o = I;
        // Compute second term: (∂e_tmin / ∂e_o) * e_d (3x3 * 3x1 = 3x3)
        dp_de_o += glm::outerProduct(e_d, etmin_de_o);
        // Compute third term: e_tmin * (∂e_d / ∂e_o)  (scalar * 3x3 = 3x3)
        dp_de_o += etmin * Jed_eo;
        // 2.4) Derivatives for the pinhole projection
        // (d) project p -> (u,v).  Then chain:
        //     d(u,v)/de_o = J_{(u,v),p} * dp/de_o
        float px_camera = hitCam.x;
        float py_camera = hitCam.y;
        float pz_camera = hitCam.z;
        // Compute derivatives
        float inv_pz = 1.0f / pz_camera;
        float inv_pz2 = inv_pz * inv_pz; // 1/pz^2
        // Construct Jacobian matrix J_(u,v),p (2x3)
        glm::mat2x3 J_uv_p;
        J_uv_p[0][0] = fx * inv_pz; // ∂u/∂px
        J_uv_p[0][1] = 0.0f; // ∂u/∂py
        J_uv_p[0][2] = -fx * px_camera * inv_pz2; // ∂u/∂pz

        J_uv_p[1][0] = 0.0f; // ∂v/∂px
        J_uv_p[1][1] = fy * inv_pz; // ∂v/∂py
        J_uv_p[1][2] = -fy * py_camera * inv_pz2; // ∂v/∂pz

        // PSEUDO: Suppose we do it in world space
        // build the relevant Jacobians:

        // 5) chain them: J_uv_eo = J_uv_p * dp_de_o => (2×3) * (3×3) = 2×3
        glm::mat2x3 J_uv_eo(0.0f);

        J_uv_eo = multiply2x3_3x3(J_uv_p, glm::transpose(dp_de_o)); // Transpose due to column-major memory layout

        // 6) Multiply that by dLoss/dI if your pixel's intensity is
        //    "I[u,v] += emissionPower" or something similar.
        //    If you have dLoss/d(u,v) as well, you would incorporate that,
        //    but let's assume your gradientImage is effectively dL/dI.

        // We want dL/d(e_o) = dLoss_dI * dI/d(e_o).
        // If "I" is just the 1-pixel deposit, then dI/d(u,v) ~ 1 in your discrete sense.
        // => dL/d(e_o) = dLoss_dI * [ ∂u/∂e_o , ∂v/∂e_o ] basically.
        // The result is a 1×3 vector. We can combine the two rows:

        // row0 = partial u / partial e_o
        glm::vec3 dU_deo = {J_uv_eo[0][0], J_uv_eo[0][1], J_uv_eo[0][2]};
        // row1 = partial v / partial e_o
        glm::vec3 dV_deo = {J_uv_eo[1][0], J_uv_eo[1][1], J_uv_eo[1][2]};
        float& dx_u = dU_deo.x;
        float& dy_u = dU_deo.y;
        float& dz_u = dU_deo.z;
        dz_u = 0;
        float dx_v = dV_deo.x;
        float dy_v = dV_deo.y;
        float dz_v = dV_deo.z;
        // If the photon’s contribution to the pixel is direct =>
        //   dI/d(u,v) = 1,
        //   so dL/d(e_o) = dLoss_dI * [ dU_deo + dV_deo ]
        // or you might choose to handle them separately:
        //glm::vec3 dL_deo = dU_deo * dLoss * emissionPower;

        */
        glm::mat2x3 multiply2x3_3x3(const glm::mat2x3 &A, const glm::mat3 &B) const {
            glm::mat2x3 result;
            // Manual matrix multiplication
            result[0][0] = A[0][0] * B[0][0] + A[1][0] * B[0][1] + A[2][0] * B[0][2];
            result[1][0] = A[0][0] * B[1][0] + A[1][0] * B[1][1] + A[2][0] * B[1][2];
            result[2][0] = A[0][0] * B[2][0] + A[1][0] * B[2][1] + A[2][0] * B[2][2];

            result[1][1] = A[1][0] * B[0][0] + A[1][1] * B[0][1] + A[1][2] * B[0][2];
            result[1][1] = A[1][0] * B[1][0] + A[1][1] * B[1][1] + A[1][2] * B[1][2];
            result[1][1] = A[1][0] * B[2][0] + A[1][1] * B[2][1] + A[1][2] * B[2][2];

            return result;
        }


        float bilinearSample(const float *image, int width, int height, float x, float y) const {
            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // Clamp to image boundaries
            x0 = std::clamp(x0, 0, width - 1);
            y0 = std::clamp(y0, 0, height - 1);
            x1 = std::clamp(x1, 0, width - 1);
            y1 = std::clamp(y1, 0, height - 1);

            // Compute interpolation weights
            float dx = x - x0;
            float dy = y - y0;

            // Fetch pixel values
            float I00 = image[y0 * width + x0];
            float I10 = image[y0 * width + x1];
            float I01 = image[y1 * width + x0];
            float I11 = image[y1 * width + x1];

            // Bilinear interpolation formula
            return (1 - dx) * (1 - dy) * I00 + dx * (1 - dy) * I10 +
                   (1 - dx) * dy * I01 + dx * dy * I11;
        }

        glm::vec3 sampleDirectionTowardAperture(
            const glm::vec3 &lightPos,
            const glm::vec3 &apertureCenter,
            const glm::vec3 &apertureNormal,
            glm::vec3 &apertureHitpoint,
            float apertureRadius,
            uint64_t photonID) const {
            // pick random point on the lens
            apertureHitpoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
            // direction from light to lens point
            glm::vec3 dir = apertureHitpoint - lightPos;
            return normalize(dir);
        }

        glm::vec3 samplePointOnDisk(size_t photonID,
                                    const glm::vec3 &center,
                                    const glm::vec3 &normal,
                                    float radius) const {
            // Or use any 2D disk sampling approach (e.g., concentric disk sampling).
            // We'll do a simple naive approach:
            float r = radius * sqrt(m_rng[photonID].nextFloat());
            float theta = 2.f * M_PI * m_rng[photonID].nextFloat();


            glm::vec3 refVec = sampleRandomDirection(photonID);
            // Ensure refVec is not parallel or nearly parallel to normal
            if (abs(dot(normal, refVec)) > 0.999f) {
                refVec = glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)); // Use a fixed backup vector
            }
            // Construct orthonormal basis for the disk plane
            glm::vec3 u = normalize(cross(normal, refVec));
            glm::vec3 v = cross(normal, u);

            float dx = r * cos(theta);
            float dy = r * sin(theta);

            glm::vec3 offset = dx * u + dy * v;
            return center + offset;
        }

        bool checkCameraPlaneIntersection(
            const glm::vec3 &rayOriginWorld,
            const glm::vec3 &rayDirWorld,
            glm::vec3 &hitPointCam, // out: intersection in camera space
            float &tIntersect, // out: parameter t
            float &contributionScore // out: parameter contributionScore
        ) const {
            // 1) Transform to camera space

            glm::mat4 entityTransform = m_cameraTransform->getTransform();
            // Camera plane normal in world space
            glm::vec3 cameraPlaneNormalWorld = glm::normalize(
                glm::mat3(entityTransform) * glm::vec3(0.0f, 0.0f, -1.0f));
            glm::vec3 cameraPlanePointWorld = glm::vec3(
                entityTransform *
                glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)); // A point on the plane

            // Ray-plane intersection

            // Ray-plane intersection calculation
            float denom = glm::dot(cameraPlaneNormalWorld, rayDirWorld);
            if (std::abs(denom) < 1e-6) {
                return false; // Ray is parallel to the plane
            }
            glm::vec3 p0l0 = cameraPlanePointWorld - rayOriginWorld;
            float t = glm::dot(p0l0, cameraPlaneNormalWorld) / denom;
            if (t < 1e-6) {
                return false; // Intersection is behind the ray origin
            }
            glm::vec3 intersectionPoint = rayOriginWorld + t * rayDirWorld;

            glm::mat4 view = glm::inverse(entityTransform);
            glm::vec4 intersectionPointCamera = view * glm::vec4(intersectionPoint, 1.0f);
            glm::vec3 intersectionCamSpace = glm::vec3(intersectionPointCamera) / intersectionPointCamera.w;

            // Sensor plane bounds in camera space
            float halfW = (m_camera->parameters().width * 0.5f) / m_camera->parameters().fx;
            float halfH = (m_camera->parameters().height * 0.5f) / m_camera->parameters().fy;

            // Check bounds
            if (intersectionCamSpace.x < -halfW || intersectionCamSpace.x > halfW ||
                intersectionCamSpace.y < -halfH || intersectionCamSpace.y > halfH) {
                return false; // Outside sensor bounds
            }

            float cosAngle = std::abs(glm::dot(cameraPlaneNormalWorld, rayDirWorld)); // Ensure positive cosine
            contributionScore = cosAngle; // Higher cosine means closer to perpendicular


            // If we reach here, the ray intersects the camera plane within bounds
            hitPointCam = intersectionPoint; // Intersection point in camera space
            tIntersect = t; // Distance along the ray to the intersection
            return true;
        }


        // ---------------------------------------------------------------------
        //  accumulateOnSensor
        // ---------------------------------------------------------------------
        void accumulateOnSensor(size_t photonID, const glm::vec3 &hitPointCam, float photonFlux) const {
            // 2. Project to the image plane using pinhole intrinsics:
            // Important: Z_cam should be > 0 for a point in front of the camera.
            //

            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;
            float X = hitPointCam.x;
            float Y = hitPointCam.y;
            float Z = hitPointCam.z;
            float xPixel = (fx * X / Z) + cx;
            float yPixel = (fy * Y / Z) + cy;
            // 2. Determine the neighboring pixels.
            // Compute the lower-left (floor) pixel coordinates and the fractional offsets.
            int x0 = static_cast<int>(std::floor(xPixel));
            int y0 = static_cast<int>(std::floor(yPixel));
            float dx = xPixel - x0;
            float dy = yPixel - y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // 3. Compute the bilinear weights for the 4 pixels.
            float w00 = (1.0f - dx) * (1.0f - dy); // weight for pixel at (x0, y0)
            float w10 = dx * (1.0f - dy); // weight for pixel at (x1, y0)
            float w01 = (1.0f - dx) * dy; // weight for pixel at (x0, y1)
            float w11 = dx * dy; // weight for pixel at (x1, y1)

            // 4. Pre-correct the photon flux with gamma adjustment.
            float correctedFlux = std::pow(photonFlux, 1.0f / m_gpuData.renderInformation->gamma);

            // 5. Retrieve image dimensions.
            const size_t imageWidth = m_camera->parameters().width;
            const size_t imageHeight = m_camera->parameters().height;

            // Helper lambda to add the weighted flux contribution to a given pixel.
            auto addFluxToPixel = [&](int px, int py, float weight) {
                // Only update if the pixel is inside the image bounds.
                if (px >= 0 && px < static_cast<int>(imageWidth) &&
                    py >= 0 && py < static_cast<int>(imageHeight)) {
                    size_t pixelIndex = static_cast<size_t>(py) * imageWidth + static_cast<size_t>(px);
                    float fluxToAdd = weight * correctedFlux;

                    // Use atomic operations to safely update the pixel value.
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                            imageMemoryAtomic(m_gpuData.imageMemory[pixelIndex]);

                    // Optionally, prevent saturation by clamping the pixel value to 1.0f.
                    float currentValue = imageMemoryAtomic.load();
                    float newValue = std::min(1.0f, currentValue + fluxToAdd);
                    fluxToAdd = newValue - currentValue; // Adjust flux to the remaining margin.
                    imageMemoryAtomic.fetch_add(fluxToAdd);
                }
            };
            // 6. Distribute the corrected flux into the four neighboring pixels.
            addFluxToPixel(x0, y0, w00);
            addFluxToPixel(x1, y0, w10);
            addFluxToPixel(x0, y1, w01);
            addFluxToPixel(x1, y1, w11);
            // 7. Atomically update the photon count.
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
                    photonsAccumulatedAtomic(m_gpuData.renderInformation->photonsAccumulated);
            photonsAccumulatedAtomic.fetch_add(static_cast<uint64_t>(1));
        }

        // ---------------------------------------------------------------------
        //  Helper: sample an emissive gaussian object
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveGaussian(size_t photonID, size_t &entityID) const {
            // Simple Linear Congruential Generator (LCG) for RNG
            std::array<size_t, 10> samples{}; // TODo max 10 light sources supported currently
            size_t i = 0;
            for (size_t entityIdx = 0; entityIdx < m_gpuData.numGaussians; ++entityIdx) {
                if (m_gpuData.gaussianInputAssembly[entityIdx].emission > 0.f) {
                    samples[i] = entityIdx;
                    i++;
                }
            }
            entityID = 0;
            // Select a random Gaussian with emissive properties
            return samples[m_rng[photonID].nextUInt() % i];
        }

        // ---------------------------------------------------------------------
        //  sampleGaussianPositionAndNormal
        // ---------------------------------------------------------------------
        void sampleGaussianPositionAndNormal(size_t entityID, size_t emissiveEntityIdx,
                                             size_t photonID,
                                             glm::vec3 &outPos,
                                             glm::vec3 &outNormal,
                                             float &emissionPower) const {
            const GaussianInputAssembly &gaussian = m_gpuData.gaussianInputAssembly[emissiveEntityIdx];
            // ------------------------------------------------------------------
            // 1. Prepare the normal, find two tangent vectors for the plane.
            // ------------------------------------------------------------------
            glm::vec3 n = glm::normalize(gaussian.normal);
            glm::vec3 t1;
            glm::vec3 t2;
            buildTangentBasis(n, t1, t2);
            // 2. Repeatedly draw samples from a standard normal, then scale
            //    them by (sigma_x, sigma_y), until they fall inside the ellipse.

            float x, y; // final offsets in local 2D coords

            // (a) Generate two uniform randoms in [0,1)
            float u1 = m_rng[photonID].nextFloat();
            float u2 = m_rng[photonID].nextFloat();

            // (b) Box-Muller transform for standard normal
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * M_PIf * u2;
            float z0 = r * cosf(theta); // ~ N(0,1)
            float z1 = r * sinf(theta); // ~ N(0,1)

            // (c) Scale by anisotropic stddev (sigma_x, sigma_y)
            x = z0 * gaussian.scale.x;
            y = z1 * gaussian.scale.y;

            // (d) Check elliptical boundary
            //     If scale.x=1 => maximum distance is 1 meter in X
            //     If scale.y=1 => maximum distance is 1 meter in Y
            //     For ellipse: (x/σx)^2 + (y/σy)^2 <= 1
            float ellipseParam = (x * x) / (gaussian.scale.x * gaussian.scale.x)
                                 + (y * y) / (gaussian.scale.y * gaussian.scale.y);

            // ------------------------------------------------------------------
            // 3. Offset the center by (x, y) in the plane spanned by (t1, t2).
            // ------------------------------------------------------------------
            glm::vec3 offset = x * t1 + y * t2;
            outPos = gaussian.position + offset;

            // Normal remains the same as the Gaussian's normal
            outNormal = n;

            // Emission power (or flux) from your stored value
            // ------------------------------------------------------------------
            // 4. Compute Emission Power per Sample
            // ------------------------------------------------------------------
            // Total emission power from the Gaussian
            float P_total = gaussian.emission;

            // Compute the Gaussian PDF at the sampled (x, y)
            float sigma_x = gaussian.scale.x;
            float sigma_y = gaussian.scale.y;

            // Gaussian PDF (unnormalized since we are within the ellipse)
            float gaussianPDF = (1.0f / (2.0f * M_PIf * sigma_x * sigma_y)) *
                                expf(-0.5f * ((x * x) / (sigma_x * sigma_x) + (y * y) / (sigma_y * sigma_y)));

            // Area of the ellipse
            float ellipseArea = M_PIf * sigma_x * sigma_y;

            // Since we are using rejection sampling, the samples are uniformly distributed over the ellipse
            // Probability density of the uniform distribution over the ellipse
            float uniformPDF = 1.0f / ellipseArea;

            // Weight for the sample based on the ratio of Gaussian PDF to uniform PDF
            // This ensures that emissionPower reflects the importance of the sample
            float weight = gaussianPDF / uniformPDF; // = (1 / (2πσxσy)) * exp(...) / (1 / πσxσy) = 0.5 * exp(...)

            // Emission power per sample
            // Distribute P_total across samples based on weight
            emissionPower = P_total * weight;
        }


        // ---------------------------------------------------------------------
        //  randomUnitVector using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 randomUnitVector(size_t photonID) const {
            float theta = m_rng[photonID].nextFloat() * 2.0f * M_PI; // [0, 2π)
            float z = m_rng[photonID].nextFloat() * 2.0f - 1.0f; // [-1, 1)
            float r = sqrtf(1.0f - z * z); // Radius at z

            float x = r * cosf(theta);
            float y = r * sinf(theta);

            return glm::vec3(x, y, z); // Already normalized
        }


        // Constructs an orthonormal basis (T, B, N) given a normal N.
        static void buildTangentBasis(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B) {
            // Any vector not collinear with N will do for "temp"
            glm::vec3 temp = (fabs(N.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);

            T = glm::normalize(glm::cross(temp, N));
            B = glm::cross(N, T);
            // Now T, B, N is an orthonormal basis
        }

        glm::vec3 sampleCosineWeightedHemisphere(
            const glm::vec3 &normal,
            size_t photonID) // random [0,1]
        const {
            // Step 1: Convert to spherical coords for cosine-weighted distribution
            float u1 = m_rng[photonID].nextFloat();
            float u2 = m_rng[photonID].nextFloat();
            float r = std::sqrt(u1);
            float phi = 2.0f * M_PIf * u2; // M_PIf = float version of pi

            // Step 2: Local coordinates (z up)
            float x = r * std::cos(phi);
            float y = r * std::sin(phi);
            float z = std::sqrt(1.0f - u1);

            // Step 3: Build a local orthonormal basis around 'normal'
            glm::vec3 t, b;
            buildTangentBasis(normal, t, b);

            // Step 4: Transform from local [x, y, z] into world space
            glm::vec3 sampleWorld = x * t + y * b + z * normal;
            return glm::normalize(sampleWorld);
        }


        // ---------------------------------------------------------------------
        //  sampleRandomHemisphere (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomHemisphere(const glm::vec3 &normal, size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            if (glm::dot(r, normal) < 0.f) {
                r = -r;
            }
            return glm::normalize(r);
        }

        // ---------------------------------------------------------------------
        //  sampleRandomDirection (Lambertian reflection) using PCG32
        // ---------------------------------------------------------------------
        glm::vec3 sampleRandomDirection(size_t photonID) const {
            glm::vec3 r = randomUnitVector(photonID);
            return glm::normalize(r);
        }
    };
}

#endif //PATHTRACER2DGSKERNELBACKWARD_H
