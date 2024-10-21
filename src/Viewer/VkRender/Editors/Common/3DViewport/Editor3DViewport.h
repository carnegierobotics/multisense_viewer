//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
#define MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H

#include <Viewer/VkRender/Editors/Video/VideoPlaybackSystem.h>

#include "Viewer/VkRender/Editors/PipelineManager.h"
#include "Viewer/VkRender/Editors/Editor.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Editors/Common/SceneRenderer.h"

namespace VkRender {

    class Editor3DViewport : public Editor {
    public:
        Editor3DViewport() = delete;

        explicit Editor3DViewport(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;


        void onSceneLoad(std::shared_ptr<Scene> scene) override;
        void onEditorResize() override;

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;



    private:
        Camera m_editorCamera;
        std::shared_ptr<Scene> m_activeScene;

        SceneRenderer* m_sceneRenderer;

    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
