//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DViewport.h"
#include "Editor3DLayer.h"

#include "Viewer/VkRender/Components/MaterialComponent.h"
#include "Viewer/VkRender/Editors/Common/CommonEditorFunctions.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Components/Components.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/MeshComponent.h"

namespace VkRender {
    Editor3DViewport::Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUI("Editor3DLayer");
        addUIData<Editor3DViewportUI>();
        m_sceneRenderer = reinterpret_cast<SceneRenderer *>(m_context->addSceneRendererWithUUID(uuid));
    }

    void Editor3DViewport::onEditorResize() {
        m_editorCamera.setPerspective(static_cast<float>(m_createInfo.width) / m_createInfo.height);
    }

    void Editor3DViewport::onSceneLoad(std::shared_ptr<Scene> scene) {

        m_editorCamera = Camera(m_createInfo.width, m_createInfo.height);
        m_activeScene = m_context->activeScene();

    }

    void Editor3DViewport::onUpdate() {
        m_activeScene = m_context->activeScene();
        if (!m_activeScene)
            return;

        m_sceneRenderer->update();
        auto& camera = m_sceneRenderer->getCamera();
        camera = m_editorCamera;
    }


    void Editor3DViewport::onRender(CommandBuffer &commandBuffer) {
        //m_sceneRenderer->render(commandBuffer);
    }

    void Editor3DViewport::onMouseMove(const MouseButtons &mouse) {
        if (ui()->hovered && mouse.left && !ui()->resizeActive) {
            m_editorCamera.rotate(mouse.dx, mouse.dy);
        }
    }

    void Editor3DViewport::onMouseScroll(float change) {
        if (ui()->hovered)
            m_editorCamera.setArcBallPosition((change > 0.0f) ? 0.95f : 1.05f);
    }
};
