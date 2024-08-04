//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORVIEWPORT_H
#define MULTISENSE_VIEWER_EDITORVIEWPORT_H

#include "Viewer/VkRender/Editor.h"
#include "Viewer/Scenes/Default/DefaultScene.h"

namespace VkRender {

    class EditorViewport : public Editor {
    public:
        EditorViewport() = delete;

        explicit EditorViewport(VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            addUI("EditorUILayer");
            addUI("DebugWindow");

            // Grid and objects
            m_scene = std::make_unique<DefaultScene>(*m_context);
        }

        void onUpdate() override {
            m_scene->update();
        }

        void onRender(CommandBuffer& drawCmdBuffers) override;

        std::unique_ptr<DefaultScene> m_scene;
    };
}

#endif //MULTISENSE_VIEWER_EDITORVIEWPORT_H
