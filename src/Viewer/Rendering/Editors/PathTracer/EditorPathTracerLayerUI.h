//
// Created by magnus on 1/2/25.
//

#ifndef MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
#define MULTISENSE_VIEWER_PATHTRACERLAYERUI_H

#include "Viewer/Rendering/ImGui/Layer.h"

#include "Viewer/Rendering/Editors/EditorIncludes.h"

namespace VkRender {

    struct EditorPathTracerLayerUI : EditorUI {


        // Constructor that copies everything from base EditorUI
        EditorPathTracerLayerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}
    };

}
#endif //MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
