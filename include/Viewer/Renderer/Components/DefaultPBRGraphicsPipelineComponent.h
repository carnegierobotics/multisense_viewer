//
// Created by magnus on 4/12/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H
#define MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H

#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/Core/CommandBuffer.h"
#include "Viewer/Renderer/Components/GLTFModelComponent.h"

namespace RenderResource {

    struct DefaultPBRGraphicsPipelineComponent {
        DefaultPBRGraphicsPipelineComponent() = default;

        DefaultPBRGraphicsPipelineComponent(const DefaultPBRGraphicsPipelineComponent &) = default;

        DefaultPBRGraphicsPipelineComponent &
        operator=(const DefaultPBRGraphicsPipelineComponent &other) { return *this; }

        DefaultPBRGraphicsPipelineComponent(VkRender::RenderUtils *utils,
                                            const VkRender::GLTFModelComponent &modelComponent) {

        }

        VkPipeline pipeline = VK_NULL_HANDLE;

        void draw(CommandBuffer *commandBuffer, uint32_t cbIndex);
    };

}
#endif //MULTISENSE_VIEWER_DEFAULTPBRGRAPHICSPIPELINECOMPONENT_H
