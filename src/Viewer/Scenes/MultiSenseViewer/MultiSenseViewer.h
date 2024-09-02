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

    struct ColmapCamera {
        int cameraId;
        std::string modelName;
        int width;
        int height;
        double focalLength;
        double cx, cy;
        double k1, k2;  // Radial distortion coefficients
    };


    class MultiSenseViewer : public Scene {

    public:
        explicit MultiSenseViewer(Renderer& ctx, const std::string& name);

        void update(uint32_t i) override;

        ~MultiSenseViewer() override{
        }

        std::vector<ColmapCameraPose> loadColmapImages(const std::string &filePath);
        void applyColmapCameraPoses(const std::vector<ColmapCameraPose> &cameraPoses, double d);

        ColmapCamera loadColmapCamera(const std::string &filePath, int targetCameraId);

        double computeFOV(double focalLength, double sensorSize);
    };
}

#endif //MULTISENSE_VIEWER_MULTISENSEVIEWER_H
