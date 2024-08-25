//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEVIEWER_H
#define MULTISENSE_VIEWER_MULTISENSEVIEWER_H

#include "Viewer/VkRender/Scene.h"

namespace VkRender {
    struct ColmapCameraPose {
        glm::quat rotation;
        glm::vec3 translation;
        std::string imageName;
    };
    class MultiSenseViewer : public Scene {

    public:
        explicit MultiSenseViewer(Renderer& ctx);

        void update(uint32_t i) override;

        ~MultiSenseViewer() override{
        }

        std::vector<ColmapCameraPose> loadColmapCameras(const std::string &filePath);
        void applyColmapCameraPoses(const std::vector<ColmapCameraPose> &cameraPoses);
    };
}

#endif //MULTISENSE_VIEWER_MULTISENSEVIEWER_H
