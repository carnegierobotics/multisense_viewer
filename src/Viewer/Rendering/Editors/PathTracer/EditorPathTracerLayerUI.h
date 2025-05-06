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
        explicit EditorPathTracerLayerUI(const EditorUI &baseUI) : EditorUI(baseUI) {}

        bool    reloadRenderer   = false;
        bool    render           = false;
        bool    renderToViewport = true;

        SYCLDeviceType selectedDevice = SYCLDeviceType::CPU;
        int         photonCount    = 1000;  // default photon count
        int     photonExponent   = 2;     // 10^6 = 1 000 000 default



    };

}
#endif //MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
