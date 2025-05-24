#ifndef QUADRICCOLLECTIONCOMPONENT_H
#define QUADRICCOLLECTIONCOMPONENT_H

#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "tinyply.h" // Assumes tinyply is available
#include "Viewer/Scenes/Entity.h"

namespace VkRender {
    class QuadricCollectionComponent {
    public:
        std::filesystem::path filePath;
        // Member arrays for each quadric
        std::vector<glm::vec3> positions; // Translation (center) in world space
        std::vector<glm::quat> rotations; // Orientation (rotation to align local +Z to surface normal)
        std::vector<float> a; // Quadric parameter a
        std::vector<float> b; // Quadric parameter b
        std::vector<float> c; // Quadric parameter c (curvature)
        std::vector<float> t_x; // Parameter t_x (for tanh sign factor)
        std::vector<float> t_y; // Parameter t_y (for tanh sign factor)
        std::vector<float> threshold; // Threshold (default 0.01)
        std::vector<float> beta; // Beta (default 0)
        std::vector<float> kernelScale; // Beta (default 0)

        std::vector<glm::vec2> min; // Beta (default 0)
        std::vector<glm::vec2> max; // Beta (default 0)


        // Resize to hold n quadrics
        void resize(size_t n) {
            positions.resize(n);
            rotations.resize(n);
            a.resize(n);
            b.resize(n);
            c.resize(n);
            t_x.resize(n);
            t_y.resize(n);
            threshold.resize(n);
            beta.resize(n);
            kernelScale.resize(n);

            min.resize(n);
            max.resize(n);
        }

        // Reserve capacity for n quadrics
        void reserve(size_t n) {
            positions.reserve(n);
            rotations.reserve(n);
            a.reserve(n);
            b.reserve(n);
            c.reserve(n);
            t_x.reserve(n);
            t_y.reserve(n);
            threshold.reserve(n);
            beta.reserve(n);
            kernelScale.reserve(n);
            min.reserve(n);
            max.reserve(n);
        }

        // Remove all quadrics
        void removeAllQuadrics() {
            positions.clear();
            rotations.clear();
            a.clear();
            b.clear();
            c.clear();
            t_x.clear();
            t_y.clear();
            threshold.clear();
            beta.clear();
            kernelScale.clear();
            min.clear();
            max.clear();
        }

        // Add a quadric with given parameters. Defaults:
        // kernelScale = 1, threshold = 0.01, beta = 0.
        void addQuadric(const glm::vec3 &position,
                        const glm::quat &rotation,
                        float a_val, float b_val, float c_val,
                        float t_x_val, float t_y_val,
                        float thresholdVal = 0.01f,
                        float betaVal = 0.0f,
                        float kernelScaleval = 0.0f,
                        glm::vec2 minVal = glm::vec2(-1.5f),
                        glm::vec2 maxVal = glm::vec2(1.5f)) {
            positions.push_back(position);
            rotations.push_back(rotation);
            a.push_back(a_val);
            b.push_back(b_val);
            c.push_back(c_val);
            t_x.push_back(t_x_val);
            t_y.push_back(t_y_val);
            threshold.push_back(thresholdVal);
            beta.push_back(betaVal);
            kernelScale.push_back(kernelScaleval);

            min.push_back(minVal);
            max.push_back(maxVal);
        }

        // Return the number of quadrics
        size_t size() const {
            return positions.size();
        }

    private:

    };
} // namespace VkRender

#endif // QUADRICCOLLECTIONCOMPONENT_H
