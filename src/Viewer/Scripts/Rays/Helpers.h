//
// Created by magnus on 3/11/25.
//

#ifndef HELPERS_H
#define HELPERS_H

namespace RayHelpers {


    static glm::vec3 computeWorldHitPoint(const glm::vec3 &e_o, const glm::vec3 &e_d,
                                          const glm::vec3 &g_c,
                                          float a, float b, float c,
                                          float tx, float ty) {

        float alpha_x = tanh(tx);
        float alpha_y = tanh(ty);
        // Since R_w2g is the identity, the local coordinates are simply:
        //   local direction: e_d,l,gt = e_d
        //   local origin:    e_o,l,gt = e_o - g_c
        glm::vec3 e_d_l = e_d;
        glm::vec3 e_o_l = e_o - g_c;

        // Compute the intersection parameters.
        // Note: In your provided printout, A turns out to be zero because the x,y components of e_d are zero.
        float A = c * (alpha_x * (e_d_l.x * e_d_l.x) / (a * a) +
                       alpha_y * (e_d_l.y * e_d_l.y) / (b * b));

        float B = c * (2.0f * alpha_x * (e_o_l.x * e_d_l.x) / (a * a) +
                       2.0f * alpha_y * (e_o_l.y * e_d_l.y) / (b * b))
                  - e_d_l.z;

        float C = c * (alpha_x * (e_o_l.x * e_o_l.x) / (a * a) +
                       alpha_y * (e_o_l.y * e_o_l.y) / (b * b))
                  - e_o_l.z;

        // In the provided code, the quadratic term A is zero (or negligible)
        // so the intersection parameter is computed as:
        float disc = (B * B) - 4 * A * C;

        // Solve for the smallest positive t (g_tmin)
        float g_tmin = 0.0f;

        if (std::fabs(A) <= std::numeric_limits<float>::epsilon()) {
            g_tmin = -C / B;
        } else {
            g_tmin = (-B + std::sqrt(disc)) / (2.0f * A);
        }

        // Compute the local hit point: g_hit,l,gt = e_d,l,gt * t + e_o,l,gt
        glm::vec3 g_hit_l = e_d_l * g_tmin + e_o_l;

        // World hit point is then given by (R_g2w * local_point + g_c).
        // Since R_g2w is the identity, we simply add g_c.
        glm::vec3 g_hit = g_hit_l + g_c;

        return g_hit;
    }

}

#endif //HELPERS_H
