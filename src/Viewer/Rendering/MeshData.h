//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_MESHDATA_H
#define MULTISENSE_VIEWER_MESHDATA_H


#include "MeshParameters.h"

#include "Components/Components.h"
#include "Viewer/Rendering/Core/RenderDefinitions.h"

namespace VkRender {
    class PLYFileMeshParameters;
    class OBJFileMeshParameters;
    class CameraGizmoPerspectiveMeshParameters;
    class CameraGizmoPinholeMeshParameters;
    class CylinderMeshParameters;
    class QuadricMeshParameters;
    class CubeMeshParameters;
    class PlaneMeshParameters;

    enum MeshDataType : uint32_t {
        EMPTY = 0,
        OBJ_FILE = 1,
        PLY_FILE = 2,
        CYLINDER = 3,
        QUADRIC = 4,
        CAMERA_GIZMO_PERSPECTIVE = 5,
        CAMERA_GIZMO_PINHOLE = 6,
        CUBE = 7,
        PLANE = 8,
        MAX_NUM_TYPES = PLANE + 1
    };

    static std::array<MeshDataType, MAX_NUM_TYPES> meshDataTypeToArray() {
        return {
            EMPTY,
            CUBE,
            PLANE,
            OBJ_FILE,
            PLY_FILE,
            CYLINDER,
            QUADRIC,
            CAMERA_GIZMO_PERSPECTIVE,
            CAMERA_GIZMO_PINHOLE,
        };
    };

    // Const char * type for easy compatibility with imgui
    static std::array<const char *, MAX_NUM_TYPES> meshDataTypeToStringArray() {
        return {
            "EMPTY",
            "CUBE",
            "PLANE",
            "CYLINDER",
            "QUADRIC",
            "OBJ_FILE",
            "PLY_FILE",
            "CAMERA_GIZMO_PERSPECTIVE",
            "CAMERA_GIZMO_PINHOLE",
        };
    };

    static std::string meshDataTypeToString(MeshDataType meshDataType) {
        switch (meshDataType) {
            case EMPTY:
                return "EMPTY";
            case CUBE:
                return "CUBE";
            case PLANE:
                return "PLANE";
            case OBJ_FILE:
                return "OBJ_FILE";
            case PLY_FILE:
                return "PLY_FILE";
            case CAMERA_GIZMO_PERSPECTIVE:
                return "CAMERA_GIZMO_PERSPECTIVE";
            case CAMERA_GIZMO_PINHOLE:
                return "CAMERA_GIZMO_PINHOLE";
            case CYLINDER:
                return "CYLINDER";
            case QUADRIC:
                return "QUADRIC";
            default:
                return "Unknown";
        }
    }

    static MeshDataType stringToMeshDataType(const std::string &modeStr) {
        if (modeStr == "EMPTY")
            return EMPTY;
        if (modeStr == "CUBE")
            return CUBE;
        if (modeStr == "PLANE")
            return PLANE;
        if (modeStr == "OBJ_FILE")
            return OBJ_FILE;
        if (modeStr == "CAMERA_GIZMO_PERSPECTIVE")
            return CAMERA_GIZMO_PERSPECTIVE;
        if (modeStr == "CAMERA_GIZMO_PINHOLE")
            return CAMERA_GIZMO_PINHOLE;
        if (modeStr == "PLY_FILE")
            return PLY_FILE;
        if (modeStr == "CYLINDER")
            return CYLINDER;
        if (modeStr == "QUADRIC")
            return QUADRIC;
        // Default case, or handle unknown input
        return EMPTY;
    }


    struct MeshData {
        std::vector<Vertex> m_vertices;
        std::vector<uint32_t> m_indices;

        std::vector<DynamicVertex> m_dynamicVertices;
        std::vector<uint32_t> m_dynamicIndices;
        bool isDirty = true;
        uint32_t version = 0;
        bool isDynamic = false;

        // Constructors
        MeshData() = default;

        // From in-memory data
        MeshData(std::vector<Vertex> vertices, std::vector<uint32_t> indices)
            : m_vertices(std::move(vertices)), m_indices(std::move(indices)) {
        }


        void generateQuadricMesh(const QuadricMeshParameters &parameters);

        void generateCylinderMesh(const CylinderMeshParameters &parameters);

        void generateCameraPinholeGizmoMesh(const CameraGizmoPinholeMeshParameters &parameters);

        void generateCameraPerspectiveGizmoMesh(const CameraGizmoPerspectiveMeshParameters &parameters);

        void generateOBJMesh(const OBJFileMeshParameters &parameters);

        void generatePLYMesh(const PLYFileMeshParameters &parameters);

        void generatePlaneMesh(const PlaneMeshParameters &plane);

        void generateCubeMesh(const CubeMeshParameters & cube_mesh_parameters);


        void computeNormals();
    };
}

#endif //MULTISENSE_VIEWER_MESHDATA_H
