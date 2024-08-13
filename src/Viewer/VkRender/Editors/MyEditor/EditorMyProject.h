//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EditorMyProject
#define MULTISENSE_VIEWER_EditorMyProject

#include "Viewer/VkRender/Editor.h"

namespace VkRender {
    class EditorMyProject : public Editor {
    public:
        explicit EditorMyProject(EditorCreateInfo &createInfo, UUID uuid = UUID());

        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onUpdate() override;

        void onSceneLoad() override;
    };
}


#endif //MULTISENSE_VIEWER_EditorMyProject
