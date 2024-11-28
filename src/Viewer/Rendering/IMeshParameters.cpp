//
// Created by magnus-desktop on 11/27/24.
//

#include "Viewer/Rendering/IMeshParameters.h"

#include "Viewer/Rendering/MeshData.h"

namespace VkRender {
    std::shared_ptr<MeshData> CylinderMeshParameters::generateMeshData() const  {
        // Generate mesh data for a cylinder
        auto meshData = std::make_shared<MeshData>();
        meshData->generateCylinderMesh(*this);
        return meshData;
    }

    std::shared_ptr<MeshData> CameraGizmoMeshParameters::generateMeshData() const  {
        // Generate mesh data for camera gizmo
        auto meshData = std::make_shared<MeshData>();
        meshData->generateCameraGizmoMesh(*this);
        return meshData;
    }

    std::shared_ptr<MeshData> OBJFileMeshParameters::generateMeshData() const {
        auto meshData = std::make_shared<MeshData>();
        meshData->generateOBJMesh(*this);
        return meshData;
    }

    std::shared_ptr<MeshData> PLYFileMeshParameters::generateMeshData() const {
        auto meshData = std::make_shared<MeshData>();
        meshData->generatePLYMesh(*this);
        return meshData;
    }
}
