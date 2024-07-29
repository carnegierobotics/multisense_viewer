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

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("EditorTestLayer");

            m_guiManager->pushLayer("DebugWindow");

        }


    };
}

#endif //MULTISENSE_VIEWER_EDITORTEST_H
