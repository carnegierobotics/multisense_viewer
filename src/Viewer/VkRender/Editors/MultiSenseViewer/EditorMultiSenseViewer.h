//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H
#define MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H

#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {
    class EditorMultiSenseViewer : public Editor {
    public:
        EditorMultiSenseViewer() = delete;

        explicit EditorMultiSenseViewer(EditorCreateInfo &createInfo) : Editor(
                createInfo) {

            if (createInfo.resizeable)
                addUI("EditorUILayer");


            for (const auto& layer : createInfo.uiLayers){
                addUI(layer);
            }


            addUI("DebugWindow");
        }

        void onRender(CommandBuffer &drawCmdBuffers) override {

        }

        void onUpdate() override {

        }
    };
}


#endif //MULTISENSE_VIEWER_EDITORMULTISENSEVIEWER_H
