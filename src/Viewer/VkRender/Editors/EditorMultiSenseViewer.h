//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H
#define MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H

#include "Viewer/VkRender/Editor.h"

namespace VkRender {
    class EditorMultiSenseViewer : public Editor {
    public:
        EditorMultiSenseViewer() = delete;

        explicit EditorMultiSenseViewer(VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            m_guiManager->pushLayer("EditorUILayer");

            m_guiManager->pushLayer("MultiSenseViewerLayer");

            m_guiManager->pushLayer("DebugWindow");

        }
    };
}


#endif //MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H
