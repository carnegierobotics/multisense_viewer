//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORTEST_H
#define MULTISENSE_VIEWER_EDITORTEST_H

#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {

    class EditorTest : public Editor {
    public:
        EditorTest() = delete;

        explicit EditorTest(EditorCreateInfo &createInfo, UUID uuid = UUID()) : Editor(
                createInfo, uuid) {
            addUI("EditorUILayer");
            addUI("EditorTestLayer");
            addUI("DebugWindow");
        }

        void onRender(CommandBuffer &drawCmdBuffers) override {

        }

        void onUpdate() override {

        }

    };
}

#endif //MULTISENSE_VIEWER_EDITORTEST_H
