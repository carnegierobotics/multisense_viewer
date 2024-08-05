//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Viewport/EditorViewport.h"

#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Components/Components.h"

namespace VkRender {

    void EditorViewport::onRender(CommandBuffer& drawCmdBuffers) {

        m_context->scene()->render(drawCmdBuffers);

    }

    void EditorViewport::onUpdate() {
        m_context->scene()->update();
    }
}
