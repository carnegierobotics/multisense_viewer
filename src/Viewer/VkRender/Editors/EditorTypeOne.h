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

        EditorTypeOne(const VkRenderEditorCreateInfo &createInfo, RenderUtils &utils, Renderer &ctx) : Editor(
                createInfo, utils, ctx) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("SideBarLayer");

        }


    };
    class EditorTypeTwo : public Editor {
    public:
        EditorTypeTwo() = delete;

        EditorTypeTwo(const VkRenderEditorCreateInfo &createInfo, RenderUtils &utils, Renderer &ctx) : Editor(
                createInfo, utils, ctx) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("MultiSenseViewerLayer");

        }


    };
}

#endif //MULTISENSE_VIEWER_EDITORTYPEONE_H
