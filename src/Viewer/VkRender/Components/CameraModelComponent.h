//
// Created by magnus on 4/24/24.
//

#ifndef MULTISENSE_VIEWER_CAMERAMODELCOMPONENT_H
#define MULTISENSE_VIEWER_CAMERAMODELCOMPONENT_H

#include "Viewer/VkRender/Core/RenderDefinitions.h"

namespace VkRender {

    struct CameraModelComponent {
        CameraModelComponent() {
            init();
        }

        CameraModelComponent(const CameraModelComponent &) = delete;

        CameraModelComponent& operator=(const CameraModelComponent&) = delete;


        UBOCamera vertices{};

        void init() {

                float a = 0.5;
                float h = 2.0;
                vertices.positions = {
                        // Base (CCW from top)
                        glm::vec4(-a, a, 0, 1.0), // D
                        glm::vec4(-a, -a, 0, 1.0), // A
                        glm::vec4(a, -a, 0, 1.0), // B
                        glm::vec4(-a, a, 0, 1.0), // D
                        glm::vec4(a, -a, 0, 1.0), // B
                        glm::vec4(a, a, 0, 1.0), // C

                        // Side 1
                        glm::vec4(-a, -a, 0, 1.0), // A
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(a, -a, 0, 1.0f), // B

                        // Side 2
                        glm::vec4(a, -a, 0, 1.0), // B
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(a, a, 0, 1.0), // C

                        // Side 3
                        glm::vec4(a, a, 0, 1.0), // C
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(-a, a, 0, 1.0), // D

                        // Side 4
                        glm::vec4(-a, a, 0, 1.0), // D
                        glm::vec4(0, 0, h, 1.0), // E
                        glm::vec4(-a, -a, 0, 1.0), // A

                        // Top indicator
                        glm::vec4(-0.4, 0.6, 0, 1.0), // D
                        glm::vec4(0.4, 0.6, 0, 1.0), // E
                        glm::vec4(0, 1.0, 0, 1.0) // A
                };
        }

    };


};

#endif //MULTISENSE_VIEWER_CAMERAMODELCOMPONENT_H
