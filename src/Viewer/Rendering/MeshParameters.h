//
// Created by magnus on 3/11/25.
//

#ifndef MESHPARAMETERS_H
#define MESHPARAMETERS_H

#include <memory>
#include <string>
#include <filesystem>
#include <glm/vec3.hpp>
#include <utility>

#include <Viewer/Application/ApplicationConfig.h>

#include "IMeshParameters.h"
#include "Editors/PinholeCamera.h"


namespace VkRender {
    class CubeMeshParameters : public IMeshParameters {
    public:
        glm::vec3 origin = glm::vec3(0.0f);
        float width = 1.0f;
        float height = 1.0f;
        float depth = 1.0f;

        void setOrigin(const glm::vec3 &origin) {
            if (this->origin != origin) {
                this->origin = origin;
                setDirty();
            }
        }

        void setSize(const float &width, const float &height, const float &depth) {
            bool changed = false;
            if (this->width != width) {
                this->width = width;
                changed = true;
            }
            if (this->height != height) {
                this->height = height;
                changed = true;
            }
            if (this->depth != depth) {
                this->depth = depth;
                changed = true;
            }
            if (changed)
                setDirty();
        }

        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "CUBE";
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class PlaneMeshParameters : public IMeshParameters {
    public:
        glm::vec3 origin = glm::vec3(0.0f);
        float size = 1.0f;

        void setOrigin(const glm::vec3 &origin) {
            if (this->origin != origin) {
                this->origin = origin;
                setDirty();
            }
        }

        void setSize(const float &size) {
            if (this->size != size) {
                this->size = size;
                setDirty();
            }
        }

        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "PLANE";
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class CylinderMeshParameters : public IMeshParameters {
    public:
        glm::vec3 origin;
        glm::vec3 direction;
        float magnitude;
        float radius = 0.05f;

        void setOrigin(const glm::vec3 &origin) {
            if (this->origin != origin) {
                this->origin = origin;
                setDirty();
            }
        }

        void setDirection(const glm::vec3 &direction) {
            if (this->direction != direction) {
                this->direction = direction;
                setDirty();
            }
        }

        void setMagnitude(const float &magnitude) {
            if (this->magnitude != magnitude) {
                this->magnitude = magnitude;
                setDirty();
            }
        }

        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "Cylinder";
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };


    class QuadricMeshParameters : public IMeshParameters {
    public:
        // Quadric parameters
        float a = 1.0f; // Scale in x
        float b = 1.0f; // Scale in y
        float c = 1.0f; // Curvature scale
        float t_x = -1.0f; // Param controlling sign in x-direction
        float t_y = 1.0f; // Param controlling sign in y-direction
        // Beta-kernel parameters
        float b_beta = 0.0f; // exponent shift
        float threshold = 0.1f; // radial kernel threshold
        float kernelScale = 1.0f;
        // Sampling parameters
        int gridResolution = 50; // number of grid points in each dimension
        glm::vec2 min = glm::vec2(-0.5f);
        glm::vec2 max = glm::vec2(0.5f);


        std::string getIdentifier() const override {
            std::ostringstream oss;
            oss << "Quadric_"
                << "a" << a << "_"
                << "b" << b << "_"
                << "c" << c << "_"
                << "tx" << t_x << "_"
                << "ty" << t_y << "_"
                << "bbeta" << b_beta << "_"
                << "kernelScale" << kernelScale << "_"
                << "thresh" << threshold;
            return oss.str();
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class CameraGizmoPinholeMeshParameters : public IMeshParameters {
    public:
        PinholeParameters parameters;

        std::string getIdentifier() const override {
            return "CameraGizmoPinhole";
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class CameraGizmoPerspectiveMeshParameters : public IMeshParameters {
    public:
        ProjectionParameters parameters;

        std::string getIdentifier() const override {
            return "CameraGizmoPerspective";
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class OBJFileMeshParameters : public IMeshParameters {
    public:
        explicit OBJFileMeshParameters(std::filesystem::path path) : path(std::move(path)) {
        }

        std::filesystem::path path;

        std::string getIdentifier() const override {
            return "OBJFileMeshParameters_" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class PLYFileMeshParameters : public IMeshParameters {
    public:
        explicit PLYFileMeshParameters(std::filesystem::path path) : path(std::move(path)) {
        }

        std::filesystem::path path;

        std::string getIdentifier() const override {
            return "PLYFileMeshParameters" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };
}

#endif //MESHPARAMETERS_H
