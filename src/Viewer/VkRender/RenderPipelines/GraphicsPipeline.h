//
// Created by magnus on 8/17/24.
//

#ifndef MULTISENSE_VIEWER_GRAPHICSPIPELINE_H
#define MULTISENSE_VIEWER_GRAPHICSPIPELINE_H

#include "Viewer/VkRender/Components/Components.h"

namespace VkRender {
    class GraphicsPipeline {
    public:
        virtual ~GraphicsPipeline() = default;

        virtual void updateTransform(const TransformComponent& transform) = 0;
        virtual void updateView(const Camera& camera) = 0;
        virtual void update(uint32_t currentFrameIndex) = 0;
        virtual void bind(MeshComponent& meshComponent) = 0;
        virtual void draw(CommandBuffer& commandBuffer) = 0;

    protected:

    };
}
#endif //MULTISENSE_VIEWER_GRAPHICSPIPELINE_H
