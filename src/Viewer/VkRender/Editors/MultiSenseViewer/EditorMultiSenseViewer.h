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

            if (createInfo.resizeable)
                addUI("EditorUILayer");


            for (const auto& layer : createInfo.uiLayers){
                addUI(layer);
            }


            addUI("DebugWindow");

        }
    };
}


#endif //MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H