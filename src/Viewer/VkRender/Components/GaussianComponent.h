//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
#define MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H

#include <tinyply.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace VkRender {


    struct GaussianComponent {
        std::vector<glm::vec3> means;         // Contiguous array for mean positions
        std::vector<glm::vec3> scales;   // Contiguous array for covariance matrices
        std::vector<float> opacities;        // Contiguous array for amplitudes
        std::vector<glm::quat> rotations;
        std::vector<glm::vec3> colors;   // Contiguous array for covariance matrices

        bool addToRenderer = false;

        GaussianComponent() = default;

        explicit GaussianComponent(std::filesystem::path pathToPly) {
            loadFromPly(pathToPly);
        }

        // Resize to hold n Gaussians
        void resize(size_t n) {
            means.resize(n);
            scales.resize(n);
            opacities.resize(n);
            rotations.resize(n);
            colors.resize(n);
        }

        // Add a Gaussian
        void addGaussian(const glm::vec3 &mean, const glm::vec3 &scale, const glm::quat &rotation, float opacity,
                         glm::vec3 color) {
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


    private:
        void loadFromPly(const std::filesystem::path &path) {
            try {
                std::ifstream fileStream(path.string(), std::ios::binary);
                if (!fileStream) {
                    throw std::runtime_error("Unable to open file.");
                }

                tinyply::PlyFile file;
                file.parse_header(fileStream);

                // Define buffers
                std::vector<float> vertices, scales, opacities, rotations, colors;

                std::shared_ptr<tinyply::PlyData> vertexData, scaleData, opacityData, rotationData, colorData;

                // Request properties
                vertexData = file.request_properties_from_element("vertex", {"x", "y", "z"});
                colorData = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
                opacityData = file.request_properties_from_element("vertex", {"opacity"});
                scaleData = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
                rotationData = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

                // Read all the data
                file.read(fileStream);

                // Extract vertices
                vertices.assign((float *) vertexData->buffer.get(),
                                (float *) vertexData->buffer.get() + vertexData->count * 3);
                scales.assign((float *) scaleData->buffer.get(),
                              (float *) scaleData->buffer.get() + scaleData->count * 3);
                rotations.assign((float *) rotationData->buffer.get(),
                                 (float *) rotationData->buffer.get() + rotationData->count * 4);
                opacities.assign((float *) opacityData->buffer.get(),
                                 (float *) opacityData->buffer.get() + opacityData->count);
                colors.assign((float *) colorData->buffer.get(),
                              (float *) colorData->buffer.get() + colorData->count * 3);

                // Process the scales with exponential
                for (size_t i = 0; i < scaleData->count; ++i) {
                    scales[i * 3 + 0] = std::exp(scales[i * 3 + 0]);
                    scales[i * 3 + 1] = std::exp(scales[i * 3 + 1]);
                    scales[i * 3 + 2] = std::exp(scales[i * 3 + 2]);
                }

                // Apply sigmoid to opacities
                for (size_t i = 0; i < opacityData->count; ++i) {
                    opacities[i] = 1.0f / (1.0f + std::exp(-opacities[i]));
                }


                // Add each Gaussian from the read data
                for (size_t i = 0; i < vertexData->count; ++i) {
                    glm::vec3 mean(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
                    glm::vec3 scale(scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]);
                    glm::quat rotation(rotations[i * 4 + 3], rotations[i * 4 + 0], rotations[i * 4 + 1],
                                       rotations[i * 4 + 2]);
                    float opacity = opacities[i];
                    glm::vec3 color(colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);

                    addGaussian(mean, scale, rotation, opacity, color);
                }
            } catch (const std::exception &e) {
                std::cerr << "Error loading PLY file: " << e.what() << std::endl;
            }
        }
    };
}
#endif //MULTISENSE_VIEWER_GAUSSIANCOMPONENT_H
