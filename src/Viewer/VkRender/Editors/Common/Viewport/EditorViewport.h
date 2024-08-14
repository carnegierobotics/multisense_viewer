//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORVIEWPORT_H
#define MULTISENSE_VIEWER_EDITORVIEWPORT_H

#include "Viewer/VkRender/Editor.h"
#include "Viewer/Scenes/Default/DefaultScene.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"

namespace VkRender {

    class EditorViewport : public Editor {
    public:
        EditorViewport() = delete;

        explicit EditorViewport(EditorCreateInfo &createInfo);
        void onUpdate() override;

        void onRender(CommandBuffer& drawCmdBuffers) override;

        void onSceneLoad() override;


        ~EditorViewport(){
            m_activeScene.reset();

        }
    private:
        std::shared_ptr<Scene> m_activeScene;
        std::vector<std::unique_ptr<DefaultGraphicsPipelineComponent>> renderPipelines;

    };
}

#endif //MULTISENSE_VIEWER_EDITORVIEWPORT_H
