//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHDATA_H
#define MULTISENSE_VIEWER_MESHDATA_H

#include <optional>

#include "Viewer/Rendering/Core/RenderDefinitions.h"

namespace VkRender {
    enum MeshDataType {
        OBJ_FILE,
        POINT_CLOUD,
        PLY_FILE,
        CAMERA_GIZMO,
        CUSTOM,
    };

    struct MeshData {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        // Constructors
        MeshData() = default;

        // From in-memory data
        MeshData(std::vector<Vertex> vertices, std::vector<uint32_t> indices)
                : vertices(std::move(vertices)), indices(std::move(indices)) {}

        // From file or procedural generation
        explicit MeshData(MeshDataType dataType, const std::filesystem::path& filePath = "") {
            switch (dataType) {
                case OBJ_FILE:
                    m_filePath = filePath;
                    meshFromObjFile();
                    break;
                case PLY_FILE:
                    m_filePath = filePath;
                    meshFromPlyFile();
                    break;
                case POINT_CLOUD:
                    meshFromPointCloud();
                    break;
                case CAMERA_GIZMO:
                    meshFromCameraGizmo();
                    break;
                case CUSTOM:
                    //generateCustomMesh();
                    break;
                default:
                    Log::Logger::getInstance()->warning("Unknown MeshDataType");
                    break;
            }
        }

        // Method to modify a vertex
        void modifyVertex(size_t index, const Vertex& newVertex) {
            if (index < vertices.size()) {
                vertices[index] = newVertex;
                isDirty = true; // Mark the mesh data as modified
            } else {
                Log::Logger::getInstance()->warning("Vertex index out of range");
            }
        }

        // Flag to indicate if the mesh is dynamic (modifiable at runtime)
        bool isDynamic = false;
        // Flag to indicate if the mesh data has been modified
        bool isDirty = false;

    private:
        std::filesystem::path m_filePath;
        void meshFromObjFile();
        void meshFromPointCloud();
        void meshFromPlyFile();
        void meshFromCameraGizmo();
        void generateCustomMesh();
    };

}

#endif //MULTISENSE_VIEWER_MESHDATA_H
