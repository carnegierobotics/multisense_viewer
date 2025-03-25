//
// Created by magnus on 3/11/25.
//

#ifndef MESHPARAMETERS_H
#define MESHPARAMETERS_H

#include <memory>
#include <string>
#include <filesystem>
#include <bits/fs_path.h>
#include <glm/vec3.hpp>
#include <utility>

#include <Viewer/Application/ApplicationConfig.h>

#include "MeshData.h"
#include "IMeshParameters.h"

#include "Editors/PinholeCamera.h"


namespace VkRender {


    class CylinderMeshParameters : public IMeshParameters {
    public:
        glm::vec3 origin;
        glm::vec3 direction;
        float magnitude;
        float radius = 0.05f;

        void setOrigin(const glm::vec3& origin) {
            if (this->origin != origin) {
                this->origin = origin;
                setDirty();
            }
        }
        void setDirection(const glm::vec3& direction) {
            if (this->direction != direction) {
                this->direction = direction;
                setDirty();
            }
        }
        void setMagnitude(const float& magnitude) {
            if (this->magnitude != magnitude) {
                this->magnitude = magnitude;
                setDirty();
            }
        }
        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "Cylinder_" + std::to_string(m_uuid);
        }

        std::shared_ptr<MeshData> generateMeshData() override;

    };

    class QuadricMeshParameters : public IMeshParameters {
    public:
        // Quadric parameters
        float a       =  1.0f;  // Scale in x
        float b       =  1.0f;  // Scale in y
        float c       =  1.0f;  // Curvature scale
        float t_x     = -1.0f; // Param controlling sign in x-direction
        float t_y     =  1.0f;  // Param controlling sign in y-direction


        // Sampling parameters
        int   gridResolution = 200;  // number of grid points in each dimension
        glm::vec2 min = glm::vec2(-2.0f);
        glm::vec2 max = glm::vec2(2.0f);


        // Beta-kernel parameters
        float b_beta      = 0.0f;   // exponent shift
        float threshold   = 0.1f;   // radial kernel threshold
        float kernelScale = 1.0f;   // normalizes the radial coordinate

        float circularity = 1.0f;

        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "Quadric" + std::to_string(m_uuid);
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class CameraGizmoPinholeMeshParameters : public IMeshParameters {
    public:
        PinholeParameters parameters;
        std::string getIdentifier() const override {
            return "CameraGizmoPinhole_" + std::to_string(m_uuid);
        }
        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class CameraGizmoPerspectiveMeshParameters : public IMeshParameters {
    public:
        ProjectionParameters parameters;
        std::string getIdentifier() const override {
            return "CameraGizmoPerspective_" + std::to_string(m_uuid);
        }
        std::shared_ptr<MeshData> generateMeshData() override;
    };
    class OBJFileMeshParameters : public IMeshParameters {
    public:
        explicit OBJFileMeshParameters(std::filesystem::path  path) : path(path) {
            std::filesystem::path assetsPath = ApplicationConfig::getInstance().getUserSetting().assetsPath;
            // Check if the provided path is relative to the assets path
            if (path.string().find(assetsPath) == 0) {
                // Compute the relative path from assetsPath
                relativeAssetPath = std::filesystem::relative(path, assetsPath);
            } else {
                // Log a warning or set relativeAssetPath to an empty path if it's not valid
                relativeAssetPath.clear();
            }
        }
        std::filesystem::path path;
        std::filesystem::path relativeAssetPath;

        std::string getIdentifier() const override {
            return "OBJFileMeshParameters_" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

    class PLYFileMeshParameters : public IMeshParameters {
    public:
        explicit PLYFileMeshParameters(std::filesystem::path  path) : path(std::move(path)) {}

        std::filesystem::path path;

        std::string getIdentifier() const override {
            return "PLYFileMeshParameters" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() override;
    };

}

#endif //MESHPARAMETERS_H
