//
// Created by magnus on 4/11/24.
//

#ifndef MULTISENSE_VIEWER_OBJMODELCOMPONENT
#define MULTISENSE_VIEWER_OBJMODELCOMPONENT


#include <vulkan/vulkan_core.h>

#include <utility>


#include "Viewer/Rendering/MeshData.h"
#include "Viewer/Rendering/IMeshParameters.h"
#include "Viewer/Rendering/MeshParameterFactory.h"

#include "Viewer/Tools/Utils.h"

namespace VkRender {
    struct MeshComponent {
        MeshComponent(MeshDataType type)
            : meshParameters(MeshParameterFactory::createMeshParameters(type)) {
        }

        MeshComponent() = default;

        MeshComponent(MeshDataType type, std::filesystem::path path) : m_meshType(type), m_modelPath(std::move(path)) {
        }

        std::filesystem::path& modelPath() { return m_modelPath; }

        MeshDataType& meshDataType() { return m_meshType; }

        VkPolygonMode& polygonMode() { return m_polygonMode; }

        std::string getCacheIdentifier() const {
            if (meshParameters) {
                return meshParameters->getIdentifier();
            }
            return "MeshComponent_NoParameters";
        }

        std::shared_ptr<IMeshParameters> meshParameters;
        std::shared_ptr<IMeshParameters> data() { return meshParameters; }
        bool isDirty = false; // TODO combine this with isDirtyFlag in MeshData
        bool dynamic = false;

    private:
        VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;
        std::filesystem::path m_modelPath;
        MeshDataType m_meshType = MeshDataType::EMPTY;

    };
};


#endif //MULTISENSE_VIEWER_OBJMODELCOMPONENT