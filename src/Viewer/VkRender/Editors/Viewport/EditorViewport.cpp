//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Viewport/EditorViewport.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/SkyboxGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Components.h"
#include "Viewer/VkRender/Components/DefaultPBRGraphicsPipelineComponent.h"
#include "Viewer/VkRender/Components/GLTFModelComponent.h"

namespace VkRender {

    void EditorViewport::onRender(CommandBuffer& drawCmdBuffers) {
        for (auto [entity, skybox, gltfComponent]: m_context->registry().view<VkRender::SkyboxGraphicsPipelineComponent, GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
            //skybox.draw(&drawCmdBuffers, *drawCmdBuffers.frameIndex);
            //gltfComponent.model->draw(drawCmdBuffers.buffers[*drawCmdBuffers.frameIndex]);
        }

        for (auto [entity, resources, gltfComponent]: m_context->registry().view<VkRender::DefaultPBRGraphicsPipelineComponent, GLTFModelComponent>(
                entt::exclude<DeleteComponent>).each()) {
                //resources.draw(&drawCmdBuffers, *drawCmdBuffers.frameIndex, gltfComponent);
        }

    }
}
