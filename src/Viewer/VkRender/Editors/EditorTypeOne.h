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

        EditorTypeOne(const VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("SideBarLayer");

        }


    };
    class EditorTypeTwo : public Editor {
    public:
        EditorTypeTwo() = delete;

        EditorTypeTwo(const VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("MultiSenseViewerLayer");

        }


    };
}

#endif //MULTISENSE_VIEWER_EDITORTYPEONE_H
