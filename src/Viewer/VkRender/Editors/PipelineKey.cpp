//
// Created by mgjer on 09/10/2024.
//
#include <functional>

#include "Viewer/VkRender/Editors/PipelineKey.h"

// Hash function for PipelineKey
// Define the hash specialization for PipelineKey



namespace VkRender {
    bool PipelineKey::operator==(const PipelineKey &other) const {
        return renderMode == other.renderMode &&
               vertexShaderName == other.vertexShaderName &&
               fragmentShaderName == other.fragmentShaderName &&
               setLayouts == other.setLayouts &&
               topology == other.topology &&
               materialPtr == other.materialPtr &&
               polygonMode == other.polygonMode;
    }
}