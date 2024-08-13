//
// Created by magnus on 5/5/24.
//

#ifndef MULTISENSE_VIEWER_RENDERBASE_H
#define MULTISENSE_VIEWER_RENDERBASE_H

#include "Viewer/VkRender/Core/CommandBuffer.h"

namespace VkRender {
    class RenderBase {
    public:
        RenderBase() = default;

        /** @brief
        // Delete copy constructors, we dont want to perform shallow copied of vulkan resources leading to double deletion.
        // If copy is necessary define custom copy constructor and use move semantics or references
        */
        RenderBase(const RenderBase &) = delete;

        RenderBase &operator=(const RenderBase &) = delete;

        virtual void draw(CommandBuffer& cmdBuffer) = 0;

        virtual void update(uint32_t currentFrame) = 0;

        virtual ~RenderBase() = default;

    private:
    };
}


#endif //MULTISENSE_VIEWER_RENDERBASE_H
