//
// Created by magnus on 1/2/25.
//

#ifndef MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
#define MULTISENSE_VIEWER_PATHTRACERLAYERUI_H

#include <Viewer/Tools/SYCLDeviceSelector.h>

#include "Viewer/Rendering/ImGui/Layer.h"

#include "Viewer/Rendering/Editors/EditorIncludes.h"

namespace VkRender {
    struct EditorPathTracerLayerUI : EditorUI {
        // Constructor that copies everything from base EditorUI
        explicit EditorPathTracerLayerUI(const EditorUI& baseUI) : EditorUI(baseUI) {
        }

        bool reloadRenderer = false;
        bool render = false;
        bool renderToViewport = true;

        SYCLDeviceType selectedDevice = SYCLDeviceType::CPU;
        int photonCount = 1000; // default photon count
        int photonExponent = 2; // 10^6 = 1 000 000 default

        // BVH stuff
        bool showBVH = false;
        int bvhLevelMax = 20;
        int bvhLevelMin = 0;
        int maxLeafSize = 0;
        // -- Info
        int currentBVHLevel = 0;
        int averageLeafSize = 0;
        int maxDepth = 30;
    };
}
#endif //MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
