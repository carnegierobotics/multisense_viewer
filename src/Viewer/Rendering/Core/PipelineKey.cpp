//
// Created by mgjer on 09/10/2024.
//
#include <functional>

#include "PipelineKey.h"

#include <cstring>

// Hash function for PipelineKey
// Define the hash specialization for PipelineKey



namespace VkRender {
    /* ───────── equality ───────── */
    bool PipelineKey::operator==(const PipelineKey& o) const
    {
        return renderMode   == o.renderMode &&
               topology     == o.topology   &&
               polygonMode  == o.polygonMode&&
               vsCRC        == o.vsCRC      &&
               fsCRC        == o.fsCRC      &&
               meshId       == o.meshId      &&
               materialFlags== o.materialFlags;
    }
}
