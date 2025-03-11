//
// Created by magnus-desktop on 11/27/24.
//

#include "Viewer/Rendering/MeshParameters.h"

#include "Viewer/Rendering/MeshData.h"

namespace VkRender {
    std::shared_ptr<MeshData> CylinderMeshParameters::generateMeshData() {
        // Generate mesh data for a cylinder
        // maintain versioning
        uint32_t version = 0;
        if (m_meshData) {
            version = m_meshData->version;
        }
        auto meshData = std::make_shared<MeshData>();
        meshData->generateCylinderMesh(*this);
        m_meshData = meshData.get();
        m_meshData->version = ++version;
        return meshData;
    }

    std::shared_ptr<MeshData> QuadricMeshParameters::generateMeshData() {
        // Generate mesh data for a cylinder
        auto meshData = std::make_shared<MeshData>();
        meshData->generateQuadricMesh(*this);
        m_meshData = meshData.get();
        return meshData;
    }

    std::shared_ptr<MeshData> CameraGizmoPerspectiveMeshParameters::generateMeshData() {
        // Generate mesh data for camera gizmo
        auto meshData = std::make_shared<MeshData>();
        meshData->generateCameraPerspectiveGizmoMesh(*this);
        m_meshData = meshData.get();
        return meshData;
    }

    std::shared_ptr<MeshData> CameraGizmoPinholeMeshParameters::generateMeshData() {
        // Generate mesh data for camera gizmo
        auto meshData = std::make_shared<MeshData>();
        meshData->generateCameraPinholeGizmoMesh(*this);
        m_meshData = meshData.get();
        return meshData;
    }

    std::shared_ptr<MeshData> OBJFileMeshParameters::generateMeshData() {
        auto meshData = std::make_shared<MeshData>();
        meshData->generateOBJMesh(*this);
        m_meshData = meshData.get();
        return meshData;
    }

    std::shared_ptr<MeshData> PLYFileMeshParameters::generateMeshData() {
        auto meshData = std::make_shared<MeshData>();
        meshData->generatePLYMesh(*this);
        m_meshData = meshData.get();
        return meshData;
    }
}
