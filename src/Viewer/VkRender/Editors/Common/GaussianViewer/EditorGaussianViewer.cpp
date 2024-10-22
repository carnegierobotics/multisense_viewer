//
// Created by magnus on 8/15/24.
//

#include "Viewer/VkRender/Editors/Common/GaussianViewer/EditorGaussianViewer.h"
#include "Viewer/VkRender/Editors/Common/GaussianViewer/EditorGaussianViewerLayer.h"

#include "Viewer/Application/Application.h"

namespace VkRender {

    EditorGaussianViewer::EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid),
                                                                                          m_syclGaussianGfx(m_deviceSelector.getQueue()) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("EditorGaussianViewerLayer");

    }

    void VkRender::EditorGaussianViewer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = m_context->activeScene();
        addUIData<EditorGaussianViewerUI>();

    }

    void EditorGaussianViewer::onEditorResize() {


    }


    void VkRender::EditorGaussianViewer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorGaussianViewerUI>(m_ui);




    }

    void VkRender::EditorGaussianViewer::onRender(CommandBuffer &drawCmdBuffers) {
        auto scene = m_context->activeScene();
        m_syclGaussianGfx.render(scene);

    }

    void VkRender::EditorGaussianViewer::onMouseMove(const VkRender::MouseButtons &mouse) {
        if (ui()->hovered && mouse.left) {
            m_activeCamera->rotate(mouse.dx, mouse.dy);
        }
        if (ui()->hovered && mouse.right) {
        }
    }

    void VkRender::EditorGaussianViewer::onMouseScroll(float change) {
    }

    void EditorGaussianViewer::onKeyCallback(const Input &input) {
        if (!m_activeCamera)
            return;
        m_activeCamera->keys.up = input.keys.up;
        m_activeCamera->keys.down = input.keys.down;
        m_activeCamera->keys.left = input.keys.left;
        m_activeCamera->keys.right = input.keys.right;

    }
}