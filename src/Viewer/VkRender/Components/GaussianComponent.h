//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
#define MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace VkRender {


    struct GaussianComponent {
        std::vector<glm::vec3> means;         // Contiguous array for mean positions
        std::vector<glm::vec3> scales;   // Contiguous array for covariance matrices
        std::vector<float> opacities;        // Contiguous array for amplitudes
        std::vector<glm::quat> rotations;
        std::vector<glm::vec3> colors;   // Contiguous array for covariance matrices

        bool addToRenderer = false;

        // Resize to hold n Gaussians
        void resize(size_t n) {
            means.resize(n);
            scales.resize(n);
            opacities.resize(n);
            rotations.resize(n);
            colors.resize(n);
        }

        // Add a Gaussian
        void addGaussian(const glm::vec3& mean, const glm::vec3& scale, const glm::quat& rotation, float opacity, glm::vec3 color) {
            means.push_back(mean);
            scales.push_back(scale);
            opacities.push_back(opacity);
            rotations.push_back(rotation);
            colors.push_back(color);

        }

        // Get the number of Gaussians
        size_t size() const {
            return means.size();
        }
    };
}
#endif //MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
