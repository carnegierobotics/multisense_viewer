//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORTYPEONE_H
#define MULTISENSE_VIEWER_EDITORTYPEONE_H

#include "Viewer/VkRender/Editor.h"

namespace VkRender {

    class EditorTypeOne : public Editor {
    public:
        EditorTypeOne() = delete;

        explicit EditorTypeOne(VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("SideBarLayer");

            m_guiManager->pushLayer("DebugWindow");

        }


    };
    class EditorTypeTwo : public Editor {
    public:
        EditorTypeTwo() = delete;

        explicit EditorTypeTwo(VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("MultiSenseViewerLayer");

            m_guiManager->pushLayer("DebugWindow");

        }
    };
}

#endif //MULTISENSE_VIEWER_EDITORTYPEONE_H
