//
// Created by magnus-desktop on 11/27/24.
//

#ifndef IMESHPARAMETERS_H
#define IMESHPARAMETERS_H

#include <memory>
#include <string>
#include <filesystem>
#include <bits/fs_path.h>
#include <glm/vec3.hpp>

namespace VkRender {
    class MeshData;

    class IMeshParameters {
    public:
        virtual ~IMeshParameters() = default;
        virtual std::string getIdentifier() const = 0;
        virtual std::shared_ptr<MeshData> generateMeshData() const = 0;
    };


    class CylinderMeshParameters : public IMeshParameters {
    public:
        glm::vec3 origin;
        glm::vec3 direction;
        float magnitude;

        std::string getIdentifier() const override {
            // Generate a unique identifier based on parameters
            return "Cylinder_";
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    class CameraGizmoMeshParameters : public IMeshParameters {
    public:
        float imageSize; // TODO implement or remove
        float focalLength = 1.0f;

        std::string getIdentifier() const override {
            return "CameraGizmo_";
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };
    class OBJFileMeshParameters : public IMeshParameters {
    public:
        OBJFileMeshParameters(const std::filesystem::path& path) : path(path) {}
        std::filesystem::path path;

        std::string getIdentifier() const override {
            return "OBJFileMeshParameters_" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    class PLYFileMeshParameters : public IMeshParameters {
    public:
        PLYFileMeshParameters(const std::filesystem::path& path) : path(path) {}

        std::filesystem::path path;

        std::string getIdentifier() const override {
            return "PLYFileMeshParameters" + path.string();
        }

        std::shared_ptr<MeshData> generateMeshData() const override;
    };

    }
#endif //IMESHPARAMETERS_H
