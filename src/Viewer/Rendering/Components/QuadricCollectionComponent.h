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
        // Member arrays for each quadric
        std::vector<glm::vec3> positions; // Translation (center) in world space
        std::vector<glm::quat> rotations; // Orientation (rotation to align local +Z to surface normal)
        std::vector<float> a; // Quadric parameter a
        std::vector<float> b; // Quadric parameter b
        std::vector<float> c; // Quadric parameter c (curvature)
        std::vector<float> t_x; // Parameter t_x (for tanh sign factor)
        std::vector<float> t_y; // Parameter t_y (for tanh sign factor)
        std::vector<float> kernelScale; // Kernel scale (default 1)
        std::vector<float> threshold; // Threshold (default 0.01)
        std::vector<float> beta; // Beta (default 0)

        // Entities for rasterizer rendering
        std::vector<Entity> entityCollection;

        // Resize to hold n quadrics
        void resize(size_t n) {
            positions.resize(n);
            rotations.resize(n);
            a.resize(n);
            b.resize(n);
            c.resize(n);
            t_x.resize(n);
            t_y.resize(n);
            kernelScale.resize(n);
            threshold.resize(n);
            beta.resize(n);
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
            kernelScale.reserve(n);
            threshold.reserve(n);
            beta.reserve(n);
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
            kernelScale.clear();
            threshold.clear();
            beta.clear();
        }

        // Add a quadric with given parameters. Defaults:
        // kernelScale = 1, threshold = 0.01, beta = 0.
        void addQuadric(const glm::vec3 &position,
                        const glm::quat &rotation,
                        float a_val, float b_val, float c_val,
                        float t_x_val, float t_y_val,
                        float kernelScaleVal = 1.0f,
                        float thresholdVal = 0.01f,
                        float betaVal = 0.0f) {
            positions.push_back(position);
            rotations.push_back(rotation);
            a.push_back(a_val);
            b.push_back(b_val);
            c.push_back(c_val);
            t_x.push_back(t_x_val);
            t_y.push_back(t_y_val);
            kernelScale.push_back(kernelScaleVal);
            threshold.push_back(thresholdVal);
            beta.push_back(betaVal);
        }

        // Load quadrics from a PLY file and append them
        void addQuadricsFromFile(const std::filesystem::path &plyFilePath) {
            loadFromPly(plyFilePath);
        }

        // Return the number of quadrics
        size_t size() const {
            return positions.size();
        }

    private:
        // Load quadrics from a ply file using tinyply.
        // Expected properties (per vertex) are:
        // "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
        // "a", "b", "c", "t_x", "t_y", "kernel_scale", "threshold", "beta"
        void loadFromPly(const std::filesystem::path &path) {
            try {
                std::ifstream fileStream(path, std::ios::binary);
                if (!fileStream.is_open())
                    throw std::runtime_error("Unable to open file: " + path.string());

                tinyply::PlyFile plyFile;
                plyFile.parse_header(fileStream);

                // Request vertex properties.
                auto vertexData = plyFile.request_properties_from_element("vertex", {"x", "y", "z"});
                auto rotationData = plyFile.request_properties_from_element("vertex", {
                                                                                "rot_0", "rot_1", "rot_2", "rot_3"
                                                                            });
                auto aData = plyFile.request_properties_from_element("vertex", {"a"});
                auto bData = plyFile.request_properties_from_element("vertex", {"b"});
                auto cData = plyFile.request_properties_from_element("vertex", {"c"});
                auto t_xData = plyFile.request_properties_from_element("vertex", {"t_x"});
                auto t_yData = plyFile.request_properties_from_element("vertex", {"t_y"});
                auto kernelScaleData = plyFile.request_properties_from_element("vertex", {"kernel_scale"});
                auto thresholdData = plyFile.request_properties_from_element("vertex", {"threshold"});
                auto betaData = plyFile.request_properties_from_element("vertex", {"beta"});

                plyFile.read(fileStream);

                size_t count = vertexData->count;
                if (rotationData->count != count || aData->count != count || bData->count != count ||
                    cData->count != count || t_xData->count != count || t_yData->count != count ||
                    kernelScaleData->count != count || thresholdData->count != count || betaData->count != count) {
                    throw std::runtime_error("Inconsistent vertex count among properties.");
                }


                std::vector<float> positionsPLY(count * 3);
                std::memcpy(positionsPLY.data(), vertexData->buffer.get(), count * 3 * sizeof(float));

                std::vector<float> rotationsPLY(count * 4);
                std::memcpy(rotationsPLY.data(), rotationData->buffer.get(), count * 4 * sizeof(float));

                std::vector<float> aPLY(count);
                std::memcpy(aPLY.data(), aData->buffer.get(), count * sizeof(float));

                std::vector<float> bPLY(count);
                std::memcpy(bPLY.data(), bData->buffer.get(), count * sizeof(float));

                std::vector<float> cPLY(count);
                std::memcpy(cPLY.data(), cData->buffer.get(), count * sizeof(float));

                std::vector<float> t_xPLY(count);
                std::memcpy(t_xPLY.data(), t_xData->buffer.get(), count * sizeof(float));

                std::vector<float> t_yPLY(count);
                std::memcpy(t_yPLY.data(), t_yData->buffer.get(), count * sizeof(float));

                std::vector<float> kernelScalePLY(count);
                std::memcpy(kernelScalePLY.data(), kernelScaleData->buffer.get(), count * sizeof(float));

                std::vector<float> thresholdPLY(count);
                std::memcpy(thresholdPLY.data(), thresholdData->buffer.get(), count * sizeof(float));

                std::vector<float> betaPLY(count);
                std::memcpy(betaPLY.data(), betaData->buffer.get(), count * sizeof(float));

                // Reserve space in our vectors.
                reserve(count);

                std::random_device rd;
                std::mt19937 gen(rd()); // Mersenne Twister RNG
                std::normal_distribution<float> dist(0.0f, 0.0f); // Mean 0, standard deviation 0.01

                // Convert and add each quadric.
                for (size_t i = 0; i < count; ++i) {

                    glm::vec3 pos(
                        positionsPLY[i * 3 + 0] + dist(gen),
                        positionsPLY[i * 3 + 1] + dist(gen),
                        positionsPLY[i * 3 + 2] + dist(gen)
                    );


                    //glm::vec3 pos(
                    //    positionsPLY[i * 3 + 0],
                    //    positionsPLY[i * 3 + 1],
                    //    positionsPLY[i * 3 + 2]
                    //    );

                    positions.push_back(pos);

                    // Assuming the order in the file is (rot_0, rot_1, rot_2, rot_3) = (w, x, y, z)
                    glm::quat quat(
                        rotationsPLY[i * 4 + 0],
                        rotationsPLY[i * 4 + 1],
                        rotationsPLY[i * 4 + 2],
                        rotationsPLY[i * 4 + 3]
                    );
                    rotations.push_back(quat);

                    a.push_back(aPLY[i]);
                    b.push_back(bPLY[i]);
                    c.push_back(cPLY[i]);
                    t_x.push_back(t_xPLY[i]);
                    t_y.push_back(t_yPLY[i]);
                    kernelScale.push_back(kernelScalePLY[i]);
                    threshold.push_back(thresholdPLY[i]);
                    beta.push_back(betaPLY[i]);
                }
                std::cout << "Successfully loaded " << count << " quadrics from " << path << std::endl;

                saveToPly("Points3D.ply");
            } catch (const std::exception &e) {
                std::cerr << "Error loading quadric PLY file: " << e.what() << std::endl;
            }
        }


        // Helper function to convert a quaternion to a normal vector.
        // It rotates the canonical unit vector (0, 0, 1) using the quaternion.
        glm::vec3 quatToNormal(const glm::quat &q) {
            return q * glm::vec3(0.0f, 0.0f, 1.0f);
        }

        // Function to save the modified vertices to a binary PLY file.
        void saveToPly(const std::filesystem::path &outputPath) {
            std::ofstream out(outputPath, std::ios::binary);
            if (!out.is_open())
                throw std::runtime_error("Unable to open file for writing: " + outputPath.string());

            size_t vertexCount = positions.size();

            // Write the header
            out << "ply\n";
            out << "format binary_little_endian 1.0\n";
            out << "element vertex " << vertexCount << "\n";
            out << "property float x\n";
            out << "property float y\n";
            out << "property float z\n";
            out << "property float nx\n";
            out << "property float ny\n";
            out << "property float nz\n";
            out << "property uchar red\n";
            out << "property uchar green\n";
            out << "property uchar blue\n";
            out << "end_header\n";

            // Write each vertex data in binary.
            for (size_t i = 0; i < vertexCount; ++i) {
                // Write modified position (x, y, z)
                float x = positions[i].x;
                float y = positions[i].y;
                float z = positions[i].z;
                out.write(reinterpret_cast<const char *>(&x), sizeof(float));
                out.write(reinterpret_cast<const char *>(&y), sizeof(float));
                out.write(reinterpret_cast<const char *>(&z), sizeof(float));

                // Convert quaternion to a normal vector by rotating (0,0,1)
                glm::vec3 normal = quatToNormal(rotations[i]);
                float nx = normal.x;
                float ny = normal.y;
                float nz = normal.z;
                out.write(reinterpret_cast<const char *>(&nx), sizeof(float));
                out.write(reinterpret_cast<const char *>(&ny), sizeof(float));
                out.write(reinterpret_cast<const char *>(&nz), sizeof(float));

                // Set a light gray color. Here we use 200 for red, green, and blue.
                unsigned char color = 200;
                out.write(reinterpret_cast<const char *>(&color), sizeof(unsigned char));
                out.write(reinterpret_cast<const char *>(&color), sizeof(unsigned char));
                out.write(reinterpret_cast<const char *>(&color), sizeof(unsigned char));
            }
            out.close();
            std::cout << "Successfully saved " << vertexCount << " vertices to " << outputPath << std::endl;
        }
    };
} // namespace VkRender

#endif // QUADRICCOLLECTIONCOMPONENT_H
