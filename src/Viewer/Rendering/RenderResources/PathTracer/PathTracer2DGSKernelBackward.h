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

        // ---------------------------------------------------------
        // Second-Bounce Photon Trace (Multi-Bounce)
        // ---------------------------------------------------------
        void traceOnePhotonSecondBounceObjectGradient(size_t photonID) const {
            GPUDataOutput::Bounce& object = m_gpuDataOutput[photonID].bounce[1];
            size_t hitObjectID = object.quadricID;
            if (hitObjectID > m_gpuData.numQuadrics) {
                return;
            }

            if (!object.hitCamera)
                return;
            glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
            glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;

            auto& quadric = m_gpuData.quadricInputAssembly[hitObjectID];
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


            GPUDataOutput::Bounce& prevBounce = m_gpuDataOutput[photonID].bounce[0];

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

            glm::vec3 f = cameraPlanePointWorld; // “focal plane” point or just known plane
            glm::vec3 f_n = cameraNormal; // plane’s normal

            // The quadric in question
            auto& quadric = m_gpuData.quadricInputAssembly[hitObjectID];

            // Grab data from the forward pass
            float u = object.pixelCoordinate.x;
            float v = object.pixelCoordinate.y;
            glm::vec3 q_hit_world = object.hitPointWorld; // the final quadric->camera intersection
            float px = object.cameraHitPointLocal.x;
            float py = object.cameraHitPointLocal.y;
            float pz = object.cameraHitPointLocal.z;

            glm::vec3 a_d = object.apertureDirection; // direction from q_hit_world -> aperture
            float a_tmin = object.cameraDirectionLength; // that intersection t
            glm::vec3 e_o = m_gpuDataOutput[photonID].emissionOrigin;
            glm::vec3 e_d = m_gpuDataOutput[photonID].emissionDirection;
            glm::vec3 e_o_local = object.quadInfo.localRayOrigin;
            glm::vec3 e_d_local = object.quadInfo.localRayDirection;

            // Quadratic info from forward pass
            auto& quadInfo = object.quadInfo;
            float A = quadInfo.A;
            float B = quadInfo.B;
            float discriminant = quadInfo.discriminant;
            int rootIndex = quadInfo.rootIndex;
            if (fabs(A) < 1e-14f || discriminant < 1e-14f || rootIndex < 1) {
                return;
            }

            // The transform for this quadric
            glm::mat4 Q2W_4x4 = quadric.transform.getTransform();
            glm::mat3 quadric2World = glm::mat3(Q2W_4x4);
            glm::mat3 world2Quadric = glm::inverse(quadric2World);

            float alpha_x = tanhf(quadric.t_x);
            float alpha_y = tanhf(quadric.t_y);

            //-----------------------------------------------------------------------
            //
            // 1) Derivatives of B,C wrt quadric center q_c
            //
            //    B = dot(...) => partial wrt e_o_local
            //    e_o_local depends on q_c via e_o_local = (world2Quadric)*(e_o) - ...
            //-----------------------------------------------------------------------
            glm::vec3 grad_B_eo;
            grad_B_eo.x = quadric.c * (2.0f * alpha_x * e_d_local.x) / (quadric.a * quadric.a);
            grad_B_eo.y = quadric.c * (2.0f * alpha_y * e_d_local.y) / (quadric.b * quadric.b);
            grad_B_eo.z = 0.0f;

            glm::vec3 grad_C_eo;
            grad_C_eo.x = quadric.c * 2.0f * alpha_x * e_o_local.x / (quadric.a * quadric.a);
            grad_C_eo.y = quadric.c * 2.0f * alpha_y * e_o_local.y / (quadric.b * quadric.b);
            grad_C_eo.z = -1.0f;

            // dB/dq_c = - (transpose(world2Quadric)) * grad_B_eo
            glm::vec3 dB_dqc = -glm::transpose(world2Quadric) * grad_B_eo;
            // dC/dq_c = ...
            glm::vec3 dC_dqc = -glm::transpose(world2Quadric) * grad_C_eo;

            //-----------------------------------------------------------------------
            //
            // 2) Derivatives of t_min wrt B, C (the root chosen)
            //
            //-----------------------------------------------------------------------
            float sqrtDisc = sqrtf(discriminant);
            float inv2A = 1.0f / (2.0f * A);
            float BoverDisc = B / sqrtDisc;
            float d_tmin_dB = 0.0f;
            float d_tmin_dC = 0.0f;

            if (rootIndex == 1) {
                // minus root =>  t = (B - sqrtDisc)/(2*A)
                // Maple expansions => d_t/dB, d_t/dC
                d_tmin_dB = -inv2A * (1.0f + BoverDisc);
                d_tmin_dC = +1.0f / sqrtDisc;
            }
            else if (rootIndex == 2) {
                // plus root => t = ( -B + sqrtDisc)/(2*A)
                d_tmin_dB = inv2A * (-1.0f + BoverDisc);
                d_tmin_dC = -1.0f / sqrtDisc;
            }
            // else return, but we already checked rootIndex above

            // Chain rule to get dtmin/dq_c
            glm::vec3 dtmin_dqc = d_tmin_dB * dB_dqc + d_tmin_dC * dC_dqc;

            //-----------------------------------------------------------------------
            //
            // 3) q_hit_local = e_o_local + q_tmin* e_d_local => derivative wrt q_c
            //    but there's also - R_w2q for the local shift, etc.
            //    In your Python code: J_qhit_qc_l = outer(e_d_l, -dtmin_eol) - R_w2q
            //    The minus sign arises from how e_o_local depends on q_c.
            //    We'll replicate the same effect with: q_hit_local(dqc) = e_d_local * dtmin_dqc - world2Quadric
            //
            //-----------------------------------------------------------------------
            // By your Maple expansions, you effectively have:
            //    J_qhitLocal_qc = outer(e_d_local, dtmin_dqc) - d(e_o_local)/dq_c
            // but d(e_o_local)/dq_c = world2Quadric * d(e_o)/d(q_c) = ...
            //
            // For clarity, here is the direct approach:
            // q_hit_local = e_o_local + t_min * e_d_local
            // => derivative wrt q_c => (d e_o_local / d q_c) + (d t_min / d q_c) e_d_local
            // but d e_o_local / d q_c = - world2Quadric (assuming e_o_local = W2Q*( e_o - q_c ) ...).
            //
            // So the net is:
            //   J_qhitLocal_qc = outer(e_d_local, dtmin_dqc) - world2Quadric
            //-----------------------------------------------------------------------
            glm::mat3 outer_edl_dtmin = glm::outerProduct(e_d_local, dtmin_dqc); // (3×3)
            glm::mat3 d_qhitLocal_dqc = outer_edl_dtmin - world2Quadric;

            //-----------------------------------------------------------------------
            //
            // 4) Convert local derivative to world derivative:
            //    q_hit_world = quadric2World * q_hit_local + q_c
            // => J_ghit_qc = quadric2World * J_qhitLocal_qc + Identity(3×3)
            //
            //-----------------------------------------------------------------------
            glm::mat3 I(1.0f);
            glm::mat3 d_ghit_dqc = quadric2World * d_qhitLocal_dqc + I;

            //-----------------------------------------------------------------------
            //
            // 5) a_d = normalize(a_c - q_hit_world)
            // => J_ad_qc = d( normalize( v_tmp ) )/ d(q_c)
            // where v_tmp = (a_c - q_hit_world).
            // The standard derivative of normalize(x):
            //   d( x / ||x|| ) = ( I/||x|| - ( x x^T )/||x||^3 ) * dx
            //
            //-----------------------------------------------------------------------
            glm::vec3 v_tmp = a_c - q_hit_world;
            float v_len = glm::length(v_tmp);
            if (v_len < 1e-14f) return; // safety
            glm::mat3 I3(1.0f);
            glm::mat3 d_unit = (I3 / v_len) - (glm::outerProduct(v_tmp, v_tmp) / (v_len * v_len * v_len));
            // chain rule => minus sign because v_tmp depends on q_hit_world => q_hit_world depends on q_c
            glm::mat3 J_ad_qc = d_unit * (-d_ghit_dqc);

            //-----------------------------------------------------------------------
            //
            // 6) a_tmin = ( (f - q_hit) · f_n ) / ( a_d · f_n )
            // => we take partial wrt q_c
            // Let n_val = (f - q_hit_world)·f_n
            //     d_val = (a_d)·f_n
            //-----------------------------------------------------------------------
            float n_val = glm::dot(f - q_hit_world, f_n);
            float d_val = glm::dot(a_d, f_n);

            // d(n_val)/d(q_c) = -(f_n^T) * J_ghit_qc
            glm::vec3 d_n = -(glm::transpose(d_ghit_dqc) * f_n);
            // d(d_val)/d(q_c) = (f_n^T) * J_ad_qc
            glm::vec3 d_d = glm::transpose(J_ad_qc) * f_n;

            float denom2 = d_val * d_val;
            glm::vec3 nabla_atmin_qc = (d_val * d_n - n_val * d_d) / denom2;

            //-----------------------------------------------------------------------
            //
            // 7) p(q_c) = q_hit_world + a_tmin * a_d
            // => J_p_qc = d_ghit_dqc + outer(a_d, nabla_atmin_qc) + a_tmin * J_ad_qc
            //
            //-----------------------------------------------------------------------
            glm::mat3 J_p_qc = d_ghit_dqc
                + glm::outerProduct(a_d, nabla_atmin_qc)
                + (a_tmin * J_ad_qc);

            //-----------------------------------------------------------------------
            //
            // 8) p_camera = w2c * p(q_c).
            // => J_pc_qc = w2c * J_p_qc
            //
            //-----------------------------------------------------------------------
            glm::mat3 J_pc_qc = w2c * J_p_qc; // shape conceptually 3×3

            //-----------------------------------------------------------------------
            //
            // 9) pinhole projection:
            //   u = fx*(px/pz) + cx
            //   v = fy*(py/pz) + cy
            // => derivative wrt p_camera = 2×3
            //-----------------------------------------------------------------------
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;

            // The forward pass's p_camera is (px,py,pz) = object.cameraHitPointLocal in camera space
            // but let's confirm that’s the same as:
            glm::vec3 p_cam(px, py, pz);
            float invZ = 1.0f / p_cam.z;

            // We'll embed the 2×3 in a 3×3, ignoring the last row:
            // J_uv_pcam = [ [fx/pz, 0, -fx*(px/(pz^2))],
            //               [0,     fy/pz, -fy*(py/(pz^2))],
            //               [0,      0,          0        ] ]
            glm::mat3 J_uv_pcam(0.0f);
            J_uv_pcam[0][0] = fx * invZ; // partial of u wrt x_cam
            J_uv_pcam[0][1] = 0.0f;
            J_uv_pcam[0][2] = -fx * (p_cam.x * invZ * invZ); // partial of u wrt z_cam

            J_uv_pcam[1][0] = 0.0f;
            J_uv_pcam[1][1] = fy * invZ;
            J_uv_pcam[1][2] = -fy * (p_cam.y * invZ * invZ);

            // Multiply:  (2×3) = (2×3) * (3×3). We'll do it as a 3×3 but only first 2 rows matter.
            // Let’s call it: J_uv_qc = J_uv_pcam * J_pc_qc
            glm::mat3 J_uv_qc = glm::transpose(J_uv_pcam) * J_pc_qc;
            // Logically that’s “2×3”, but we’re storing in a 3×3 with row #2 = zero.

            //-----------------------------------------------------------------------
            //
            // 10) Next, we do the Beta kernel derivative in local quadric coords
            //     We'll replicate your python code’s steps: J_beta_uv = J_beta_xy @ J_xy_uv
            //     then J_Iuv_qc = J_beta_uv @ J_uv_qc
            glm::mat4 cameraToWorld = m_cameraTransform->getTransform();
            glm::vec4 hitPointCam4 = cameraToWorld * glm::vec4(object.cameraHitPointLocal, 1.0f);
            glm::vec3 cameraRayOrigin = hitPointCam4 / hitPointCam4.w;

            glm::vec3 cameraRayDir = -a_d;
            // do geometry intersection again:
            float tmin = FLT_MAX;
            size_t hitEntity = 0;
            glm::vec3 hitPointWorld(0.0f);
            glm::vec3 hitNormalWorld(0.0f);
            float betaContribution = 0.0f;
            GPUDataOutput::QuadraticInfo quadraticInfo{};
            bool hit = geometryIntersectionQuadric(gaussianID,
                                                   cameraRayOrigin,
                                                   cameraRayDir,
                                                   hitEntity, tmin,
                                                   hitPointWorld,
                                                   hitNormalWorld,
                                                   betaContribution,
                                                   quadraticInfo);
            if (!hit) {
                return;
            }
            // local coords of that camera->quadric intersection
            glm::vec2 p_l = quadraticInfo.hitLocal;

            /*
            glm::mat3 R_i2c(-1.0f);
            glm::vec3 d_ray2 = -a_d;
            glm::vec3 d_ray = glm::vec3((u - cx) / fx, (v - cy) / fy, 1.0f);
            float d_ray_len = glm::length(d_ray);
            glm::mat3 d_norm_ray = (I / d_ray_len) - (glm::outerProduct(d_ray, d_ray) / static_cast<float>(std::pow(
                d_ray_len, 3)));

            glm::vec3 d_ray_u(1.0f / fx, 0.0f, 0.0f);
            glm::vec3 d_ray_norm_u = d_norm_ray * d_ray_u;
            glm::vec3 d_dc_u = R_i2c * d_ray_norm_u;
            glm::vec3 d_dw_u = glm::mat3(camera2World) * d_dc_u;
            glm::vec3 d_dl_u = world2Quadric * d_dw_u;

            glm::vec3 d_dray_v(0.0f, 1.0f / fy, 0.0f);
            glm::vec3 d_ray_norm_v = d_norm_ray * d_dray_v;
            glm::vec3 d_dc_v = R_i2c * d_ray_norm_v;
            glm::vec3 d_dw_v = glm::mat3(camera2World) * d_dc_v;
            glm::vec3 d_dl_v = world2Quadric * d_dw_v;

            float A_cam = quadraticInfo.A;
            float B_cam = quadraticInfo.B;
            float C_cam = quadraticInfo.C;
            float disc_sqrt_cam = sycl::sqrt(quadraticInfo.discriminant);
            // derivative wrt A_cam, B_cam for the chosen root
            float numerator = -B_cam + disc_sqrt_cam;
            float denominator = 2 * A_cam;
            float d_numerator = +((-2 * C_cam) / disc_sqrt_cam);
            float d_denominator = 2;

            float d_tmin_A = 1 / (4 * A_cam * A_cam) * (denominator * d_numerator - numerator * d_denominator);
            float d_tmin_B = 1 / (2 * A_cam) * (-1 + (B_cam / disc_sqrt_cam));

            glm::vec3 d_l = quadraticInfo.localRayDirection;
            glm::vec3 a_l = quadraticInfo.localRayOrigin;

            float d_A_u = 2 * quadric.c * (alpha_x * d_dl_u[0] * d_l[0] / (quadric.a * quadric.a) + (alpha_y * d_dl_u[1]
                * d_l[1]) / (quadric.b * quadric.b));
            float d_B_u = 2 * quadric.c * (alpha_x * d_dl_u[0] * a_l[0] / (quadric.a * quadric.a) + (alpha_y * d_dl_u[1]
                * a_l[1]) / (quadric.b * quadric.b)) - d_dl_u[2];

            float d_A_v = 2 * quadric.c * (alpha_x * d_dl_v[0] * d_l[0] / (quadric.a * quadric.a) + (alpha_y * d_dl_v[1]
                * d_l[1]) / (quadric.b * quadric.b));
            float d_B_v = 2 * quadric.c * (alpha_x * d_dl_v[0] * a_l[0] / (quadric.a * quadric.a) + (alpha_y * d_dl_v[1]
                * a_l[1]) / (quadric.b * quadric.b)) - d_dl_v[2];

            float d_tmin_u = d_tmin_A * d_A_u + d_tmin_B * d_B_u;
            float d_tmin_v = d_tmin_A * d_A_v + d_tmin_B * d_B_v;
            // similarly handle rootIndexCam == 1, etc. (omitted for brevity)
            float d_x_u = d_l[0] * d_tmin_u + tmin * d_dl_u[0];
            float d_x_v = d_l[0] * d_tmin_v + tmin * d_dl_v[0];
            float d_y_u = d_l[1] * d_tmin_u + tmin * d_dl_u[1];
            float d_y_v = d_l[1] * d_tmin_v + tmin * d_dl_v[1];

            glm::mat2 J_xy_uv = glm::mat2(glm::vec2(d_x_u, d_y_u), glm::vec2(d_x_v, d_y_v));
            // Create a 2x3 matrix to hold the first two rows:
            */

            //-----------------------------------------------------------------------
            //
            // 11) Next, we do the Beta kernel derivative in Image Coordinates
            float g_d = quadraticInfo.geodesic;
            float x_local = quadraticInfo.hitLocal.x;
            float y_local = quadraticInfo.hitLocal.y;
            float rho = quadraticInfo.rho;
            float theta = quadraticInfo.theta;
            float a_theta = quadraticInfo.a_theta;

            float p_tmp = 4;
            float exponent = p_tmp - 1.0f;
            float d_beta_dgd = -2.0f * p_tmp * g_d * std::pow((1 - g_d * g_d), exponent);

            // Compute derivatives of rho.
            float d_rho_dx = (rho != 0.0f) ? x_local / rho : 0.0f;
            float d_rho_dy = (rho != 0.0f) ? y_local / rho : 0.0f;

            // Compute derivatives of theta.
            float denom = (x_local * x_local + y_local * y_local);
            float d_theta_dx = (denom != 0.0f) ? -y_local / denom : 0.0f;
            float d_theta_dy = (denom != 0.0f) ?  x_local / denom : 0.0f;

            // Compute derivative of aTheta with respect to theta.
            // d_aTheta_dtheta = 2 * c * cos(theta) * sin(theta) * (alpha_y / (b^2) - alpha_x / (a^2))
            float d_aTheta_dtheta = 2.0f * quadric.c * std::cos(theta) * std::sin(theta) *
                                      ((alpha_y / (quadric.b * quadric.b)) - (alpha_x / (quadric.a * quadric.a)));

            // Compute dg_d/drho.
            // sqrt_term = sqrt(4 * a_theta^2 * rho^2 + 1)
            float sqrt_term = std::sqrt(4.0f * a_theta * a_theta * rho * rho + 1.0f);
            float d_gd_drho = sqrt_term;

            // Compute dg_d/daTheta; if a_theta is zero, we use a derivative of zero.
            float d_gd_daTheta = 0.0f;
            if (a_theta != 0.0f) {
                // term1 = (2 * a_theta * rho^3) / sqrt_term
                float term1 = (2.0f * a_theta * std::pow(rho, 3)) / sqrt_term;
                // term2 = - asinh(2 * a_theta * rho) / (4 * a_theta^2) + rho / (2 * a_theta * sqrt_term)
                float term2 = - (std::asinh(2.0f * a_theta * rho)) / (4.0f * a_theta * a_theta) +
                               (rho) / (2.0f * a_theta * sqrt_term);
                d_gd_daTheta = term1 + term2;
            }

            // Chain rule for dg_d/dx and dg_d/dy.
            float d_gd_dx = d_rho_dx * sqrt_term + d_theta_dx * d_gd_daTheta * d_aTheta_dtheta;
            float d_gd_dy = d_rho_dy * sqrt_term + d_theta_dy * d_gd_daTheta * d_aTheta_dtheta;

            // Finally, compute the derivatives dβ/dx and dβ/dy.
            float d_beta_dx = d_beta_dgd * d_gd_dx;
            float d_beta_dy = d_beta_dgd * d_gd_dy;

            glm::vec3 J_beta_xy = glm::vec3(d_beta_dx, d_beta_dy, 0.0f);

            glm::mat3 J_xy_qc = glm::transpose(d_qhitLocal_dqc);
            glm::vec3 J_beta_qc = J_xy_qc * J_beta_xy;


            // Store final gradient results
            int uInt = (int)std::round(u);
            int vInt = (int)std::round(v);
            if (uInt < 0 || vInt < 0 ||
                uInt >= (int)m_camera->m_parameters.width ||
                vInt >= (int)m_camera->m_parameters.height) {
                return;
                }
            size_t pixelIndex = vInt * m_camera->m_parameters.width + uInt;

            // For demonstration, put the 2D partial dβ/du, dβ/dv in gradientImageU, gradientImageV
            m_gpuData.gradientImageU[pixelIndex] = d_beta_dx;
            m_gpuData.gradientImageV[pixelIndex] = d_beta_dy;
            m_gpuData.gradientImagePerObject[pixelIndex] = static_cast<float>(hitObjectID);

            // Also store the 3D partial J_Iuv_qc, plus maybe the q_hit_world in the same mat3
            glm::mat3 tmp(0.0f);
            // First column = derivative
            tmp[0][0] = J_beta_qc.x;
            tmp[1][0] = J_beta_qc.y;
            tmp[2][0] = J_beta_qc.z;

            // Second column = q_hit_world
            tmp[0][1] = q_hit_world.x;
            tmp[1][1] = q_hit_world.y;
            tmp[2][1] = q_hit_world.z;

            // Third column left empty or used as you wish
            // tmp[0][2] = ...
            // tmp[1][2] = ...
            // tmp[2][2] = ...

            // Store in GPU data
            m_gpuData.gradientPixelCoordinates[photonID] = glm::vec2(u, v);

            m_gpuData.photonIDGradient[photonID] = tmp;

        }

        bool castContributionRay(const glm::vec3& directLightingOrigin, const glm::vec3& cameraPlaneNormalWorld,
                                 float apertureRadius, size_t photonID, float photonFlux,
                                 glm::vec3& directLightDir,
                                 glm::vec3& apertureHitPoint,
                                 glm::vec3& cameraHitPointLocal,
                                 glm::vec2& pixelCoordinates,
                                 float& camera_t
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
                const BVHNode& node = m_gpuData.bvhNodes[currentIndex];

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
            glm::vec3& e_c = gaussianPosition;
            // Camera intrinsics
            float fx = m_camera->parameters().fx;
            float fy = m_camera->parameters().fy;
            float cx = m_camera->parameters().cx;
            float cy = m_camera->parameters().cy;


            glm::vec3 f = cameraPlanePointWorld; // e.g., defined in your camera parameters
            glm::vec3 f_n = cameraNormal; // e.g., (0,0,1) if the focal plane faces +Z

            // PIXEL LOSS GROUND TRUTH
            GPUDataOutput::Bounce& object = m_gpuDataOutput[photonID].bounce[0];
            size_t hitObjectID = object.quadricID;

            if (hitObjectID > m_gpuData.numGaussians || gaussianID > m_gpuData.numGaussians || !object.hitCamera) {
                return;
            }

            auto& newHitObject = m_gpuData.gaussianInputAssembly[hitObjectID];
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
        glm::mat2x3 multiply2x3_3x3(const glm::mat2x3& A, const glm::mat3& B) const {
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


        float bilinearSample(const float* image, int width, int height, float x, float y) const {
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
            const glm::vec3& lightPos,
            const glm::vec3& apertureCenter,
            const glm::vec3& apertureNormal,
            glm::vec3& apertureHitpoint,
            float apertureRadius,
            uint64_t photonID) const {
            // pick random point on the lens
            apertureHitpoint = samplePointOnDisk(photonID, apertureCenter, apertureNormal, apertureRadius);
            // direction from light to lens point
            glm::vec3 dir = apertureHitpoint - lightPos;
            return normalize(dir);
        }

        glm::vec3 samplePointOnDisk(size_t photonID,
                                    const glm::vec3& center,
                                    const glm::vec3& normal,
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
            const glm::vec3& rayOriginWorld,
            const glm::vec3& rayDirWorld,
            glm::vec3& hitPointCam, // out: intersection in camera space
            float& tIntersect, // out: parameter t
            float& contributionScore // out: parameter contributionScore
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
        //  Helper: sample an emissive gaussian object
        // ---------------------------------------------------------------------
        size_t sampleRandomEmissiveGaussian(size_t photonID, size_t& entityID) const {
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
                                             glm::vec3& outPos,
                                             glm::vec3& outNormal,
                                             float& emissionPower) const {
            const GaussianInputAssembly& gaussian = m_gpuData.gaussianInputAssembly[emissiveEntityIdx];
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
        static void buildTangentBasis(const glm::vec3& N, glm::vec3& T, glm::vec3& B) {
            // Any vector not collinear with N will do for "temp"
            glm::vec3 temp = (fabs(N.x) > 0.9f) ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);

            T = glm::normalize(glm::cross(temp, N));
            B = glm::cross(N, T);
            // Now T, B, N is an orthonormal basis
        }

        glm::vec3 sampleCosineWeightedHemisphere(
            const glm::vec3& normal,
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
        glm::vec3 sampleRandomHemisphere(const glm::vec3& normal, size_t photonID) const {
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
