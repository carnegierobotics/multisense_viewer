//
// Created by magnus on 4/25/25.
//

#include "IMeshParameters.h"
#include "MeshData.h"

namespace VkRender{


    void IMeshParameters::setDirty() const {
        if (m_meshData) {
            m_meshData->isDirty = true;
        }
    }
}
