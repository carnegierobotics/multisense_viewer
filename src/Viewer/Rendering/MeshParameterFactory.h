//
// Created by magnus-desktop on 11/27/24.
//

#ifndef MESHPARAMETERFACTORY_H
#define MESHPARAMETERFACTORY_H

namespace VkRender {
    class MeshParameterFactory {
    public:
        static std::shared_ptr<IMeshParameters> createMeshParameters(MeshDataType type, std::filesystem::path path) {
            switch (type) {
            case CYLINDER:
                return std::make_shared<CylinderMeshParameters>();
            case CAMERA_GIZMO:
                return std::make_shared<CameraGizmoMeshParameters>();
            case OBJ_FILE:
                return std::make_shared<OBJFileMeshParameters>(path);
            case PLY_FILE:
                return std::make_shared<PLYFileMeshParameters>(path);
            default:
                return nullptr;
            }
        }
    };
}

#endif //MESHPARAMETERFACTORY_H
