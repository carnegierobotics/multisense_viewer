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
            GPUDataOutput::Bounce &object = m_gpuDataOutput[photonID].bounce[0];
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

            // The quadric in question

            auto &quadric = m_gpuData.quadricInputAssembly[hitObjectID];
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
            //if (facingCameraDot <= 0.1f || facingLightSourceDot <= 0.1f) {
            //    return;
            //}


            // Grab data from the forward pass
            float u = object.pixelCoordinate.x;
            float v = object.pixelCoordinate.y;
            glm::vec3 q_hit_world = object.hitPointWorld; // the final quadric->camera intersection
            float px = object.cameraHitPointLocal.x;
            float py = object.cameraHitPointLocal.y;
            float pz = object.cameraHitPointLocal.z;

            glm::vec3 a_d = object.apertureDirection; // direction from q_hit_world -> aperture
            float a_tmin = object.cameraDirectionLength; // that intersection t

            glm::vec3 e_o_local = object.quadInfo.localRayOrigin;
            glm::vec3 e_d_local = object.quadInfo.localRayDirection;

            // Quadratic info from forward pass
            auto &quadInfo = object.quadInfo;
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
            } else if (rootIndex == 2) {
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



            glm::vec3 d_ray = glm::vec3((u - cx) / fx, (v- cy) / fy, 1.0f);

            glm::mat4 cameraToWorld = m_cameraTransform->getTransform();
            glm::vec4 hitPointCam4 = cameraToWorld * glm::vec4(d_ray, 1.0f);
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



            //-----------------------------------------------------------------------
            //
            // 10) Next, we do the Beta kernel derivative in local quadric coords
            //     We'll replicate your python code’s steps: J_beta_uv = J_beta_xy @ J_xy_uv
            //     then J_Iuv_qc = J_beta_uv @ J_uv_qc



            // local coords of that camera->quadric intersection
            glm::vec2 p_l = quadraticInfo.hitLocal;


            glm::mat3 R_i2c(-1.0f);
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
            float d_theta_dy = (denom != 0.0f) ? x_local / denom : 0.0f;

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
                float term2 = -(std::asinh(2.0f * a_theta * rho)) / (4.0f * a_theta * a_theta) +
                              (rho) / (2.0f * a_theta * sqrt_term);
                d_gd_daTheta = term1 + term2;
            }

            // Chain rule for dg_d/dx and dg_d/dy.
            float d_gd_dx = d_rho_dx * sqrt_term + d_theta_dx * d_gd_daTheta * d_aTheta_dtheta;
            float d_gd_dy = d_rho_dy * sqrt_term + d_theta_dy * d_gd_daTheta * d_aTheta_dtheta;

            // Finally, compute the derivatives dβ/dx and dβ/dy.
            float d_beta_dx = -d_beta_dgd * d_gd_dx;
            float d_beta_dy = -d_beta_dgd * d_gd_dy;


            glm::vec2 J_beta_uv = glm::vec2(
             d_beta_dx * J_xy_uv[0][0] +  d_beta_dy * J_xy_uv[0][1],
            d_beta_dx * J_xy_uv[1][0] +  d_beta_dy * J_xy_uv[1][1]
            );

            glm::vec3 projection = glm::vec3(
                 J_beta_uv[0] * J_uv_qc[0][0] +  J_beta_uv[1] * J_uv_qc[0][1],
                J_beta_uv[0] * J_uv_qc[1][0] +  J_beta_uv[1] * J_uv_qc[1][1],
                 J_beta_uv[0] * J_uv_qc[2][0] +  J_beta_uv[1] * J_uv_qc[2][1]
                );


            glm::vec3 J_beta_xy = -glm::vec3(d_beta_dx, d_beta_dy, 0.0f);
            glm::mat3 J_xy_qc = glm::transpose(d_qhitLocal_dqc);
            glm::vec3 J_beta_qc = J_xy_qc * J_beta_xy;

            //projection = projection + J_beta_qc;
            // Store final gradient results
            int uInt = (int) std::round(u);
            int vInt = (int) std::round(v);
            if (uInt < 0 || vInt < 0 ||
                uInt >= (int) m_camera->m_parameters.width ||
                vInt >= (int) m_camera->m_parameters.height) {
                return;
            }
            size_t pixelIndex = vInt * m_camera->m_parameters.width + uInt;

            // For demonstration, put the 2D partial dβ/du, dβ/dv in gradientImageU, gradientImageV
            m_gpuData.gradientImageU[pixelIndex] = J_beta_xy.x;
            m_gpuData.gradientImageV[pixelIndex] = J_beta_xy.y;
            m_gpuData.gradientImagePerObject[pixelIndex] = static_cast<float>(hitObjectID);

            // Also store the 3D partial J_Iuv_qc, plus maybe the q_hit_world in the same mat3
            glm::mat3 tmp(0.0f);
            // First column = derivative
            tmp[0][0] = projection.x;
            tmp[1][0] = projection.y;
            tmp[2][0] = projection.z;

            // Second column = q_hit_world
            tmp[0][1] = q_hit_world.x;
            tmp[1][1] = q_hit_world.y;
            tmp[2][1] = q_hit_world.z;

            // Third column left empty or used as you wish
            tmp[0][2] = J_beta_qc.x;
            tmp[1][2] = J_beta_qc.y;
            tmp[2][2] = J_beta_qc.z;

            // Store in GPU data
            m_gpuData.gradientPixelCoordinates[photonID] = glm::vec2(u, v);

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
