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

        explicit EditorViewport(VulkanRenderPassCreateInfo &createInfo);
        void onUpdate() override;

        void onRender(CommandBuffer& drawCmdBuffers) override;

        void onSceneLoad() override;

    };
}

#endif //MULTISENSE_VIEWER_EDITORVIEWPORT_H
