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
        bool renderUntilTarget = false;
        bool resetPathTracer = false;
        bool renderToViewport = true;

        SYCLDeviceType selectedDevice = SYCLDeviceType::CPU;
        int photonCount = 1000; // default photon count
        int photonExponent = 3; // 10^6 = 1 000 000 default
        int targetPhotonExponent = photonExponent  + 1; // 10^6 = 1 000 000 default
        int numBounces = 8;
        int targetPhotonCount = 1e7; // 10 Million

        // BVH stuff
        bool showBVH = false;
        bool showTLAS = false;
        bool showBLAS = false;
        int bvhLevelMax = 20;
        int bvhLevelMin = 0;
        int maxLeafSize = 0;
        // -- Info
        int currentBVHLevel = 0;
        int averageLeafSize = 0;
        int maxDepth = 30;
        float gamma = 2.8f;
        float exposure = 4.6f; //1.8f;

        bool saveImages = false;
        std::filesystem::path saveImagePath;

        uint64_t totalPhotonsEmitted = 0;

        // Reconstruction
        std::filesystem::path gtFolderPath;
        bool renderGradient = false;
        int gradMaterialID = 0;
        bool printMaterialID = false;
    };
}
#endif //MULTISENSE_VIEWER_PATHTRACERLAYERUI_H
