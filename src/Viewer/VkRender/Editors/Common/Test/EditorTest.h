//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORTEST_H
#define MULTISENSE_VIEWER_EDITORTEST_H

#include "Viewer/VkRender/Editor.h"

namespace VkRender {

    class EditorTest : public Editor {
    public:
        EditorTest() = delete;

        explicit EditorTest(VulkanRenderPassCreateInfo &createInfo, UUID uuid = UUID()) : Editor(
                createInfo, uuid) {

            if (createInfo.resizeable)
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
