//
// Created by magnus-desktop on 11/27/24.
//

#ifndef IMESHPARAMETERS_H
#define IMESHPARAMETERS_H

#include <memory>
#include "Viewer/Rendering/Core/UUID.h"

namespace VkRender {
    class MeshData;

    class IMeshParameters {
    public:
        virtual ~IMeshParameters() = default;
        virtual std::string getIdentifier() const = 0;
        virtual std::shared_ptr<MeshData> generateMeshData() = 0;

        void setDirty() const;

    protected:
        UUID m_uuid;
        MeshData* m_meshData = nullptr;

    };

}
#endif //IMESHPARAMETERS_H
