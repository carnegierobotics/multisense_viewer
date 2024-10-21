//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
#define MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H

#include <glm/glm.hpp>
#include <vector>

namespace VkRender {


    struct GaussianComponent {
        std::vector<glm::vec3> means;         // Contiguous array for mean positions
        std::vector<glm::mat3> covariances;   // Contiguous array for covariance matrices
        std::vector<float> amplitudes;        // Contiguous array for amplitudes

        // Precomputed values
        std::vector<glm::mat3> invCovariances;  // Inverse covariance matrices
        std::vector<float> determinants;        // Determinants of the covariance matrices

        // Resize to hold n Gaussians
        void resize(size_t n) {
            means.resize(n);
            covariances.resize(n);
            amplitudes.resize(n);
            invCovariances.resize(n);
            determinants.resize(n);
        }

        // Add a Gaussian
        void addGaussian(const glm::vec3& mean, const glm::mat3& covariance, float amplitude) {
            means.push_back(mean);
            covariances.push_back(covariance);
            amplitudes.push_back(amplitude);

            // Precompute inverse covariance and determinant
            glm::mat3 invCov = glm::inverse(covariance);
            float det = glm::determinant(covariance);

            invCovariances.push_back(invCov);
            determinants.push_back(det);
        }

        // Get the number of Gaussians
        size_t size() const {
            return means.size();
        }
    };
}
#endif //MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
