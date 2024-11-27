//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEVIEWER_H
#define MULTISENSE_VIEWER_MULTISENSEVIEWER_H

#include "Viewer/Scenes/Scene.h"

namespace VkRender {
    struct ColmapCameraPose {
        glm::quat rotation;
        glm::vec3 translation;
        std::string imageName;
    };

    struct ColmapCamera {
        int cameraId;
        std::string modelName;
        int width;
        int height;
        double focalLength;
        double cx, cy;
        double k1, k2;  // Radial distortion coefficients
    };


    class MultiSenseViewer {

    };
}

#endif //MULTISENSE_VIEWER_MULTISENSEVIEWER_H
