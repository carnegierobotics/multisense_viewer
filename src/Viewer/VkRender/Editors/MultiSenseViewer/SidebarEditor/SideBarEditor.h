//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_SIDEBAREDITOR_H
#define MULTISENSE_VIEWER_SIDEBAREDITOR_H

#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {
    class SideBarEditor : public Editor {
    public:
        SideBarEditor() = delete;

        explicit SideBarEditor(EditorCreateInfo &createInfo) : Editor(
                createInfo) {

            addUI("SideBarLayer");
            addUI("DebugWindow");
        }

        void onRender(CommandBuffer &drawCmdBuffers) override {

        }

        void onUpdate() override {

        }
    private:

    };
}


#endif //MULTISENSE_VIEWER_SIDEBAREDITOR_H
