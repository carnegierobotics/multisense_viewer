//
// Created by magnus-desktop on 11/27/24.
//

#ifndef MESHPARAMETERFACTORY_H
#define MESHPARAMETERFACTORY_H

namespace VkRender {
    class MeshParameterFactory {
    public:
        static std::shared_ptr<IMeshParameters> createMeshParameters(MeshDataType type) {
            switch (type) {
            case CYLINDER:
                return std::make_shared<CylinderMeshParameters>();
            case CAMERA_GIZMO:
                return std::make_shared<CameraGizmoMeshParameters>();
                // Add cases for other mesh types...
            default:
                return nullptr;
            }
        }
    };
}

#endif //MESHPARAMETERFACTORY_H
