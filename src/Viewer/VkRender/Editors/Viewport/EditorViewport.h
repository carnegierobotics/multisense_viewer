//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORVIEWPORT_H
#define MULTISENSE_VIEWER_EDITORVIEWPORT_H

#include "Viewer/VkRender/Editor.h"

namespace VkRender {

    class EditorViewport : public Editor {
    public:
        EditorViewport() = delete;

        explicit EditorViewport(VulkanRenderPassCreateInfo &createInfo) : Editor(
                createInfo) {

            addUI("EditorUILayer");
            addUI("DebugWindow");

            // Grid and objects


        }


    };
}

#endif //MULTISENSE_VIEWER_EDITORVIEWPORT_H
