//
// Created by magnus on 5/5/24.
//

#ifndef MULTISENSE_VIEWER_RENDERBASE_H
#define MULTISENSE_VIEWER_RENDERBASE_H

#include "Viewer/VkRender/Core/CommandBuffer.h"

namespace VkRender {
    struct RenderPassInfo {
        VkSampleCountFlagBits sampleCount;
        VkRenderPass renderPass;
        uint32_t swapchainImageCount = 1;
    };

}


#endif //MULTISENSE_VIEWER_RENDERBASE_H
