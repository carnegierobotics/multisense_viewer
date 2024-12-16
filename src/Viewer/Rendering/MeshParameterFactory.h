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
            case CAMERA_GIZMO_PERSPECTIVE:
                return std::make_shared<CameraGizmoPerspectiveMeshParameters>();
            case CAMERA_GIZMO_PINHOLE:
                return std::make_shared<CameraGizmoPinholeMeshParameters>();
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
