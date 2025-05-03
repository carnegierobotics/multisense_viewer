//
// Created by magnus on 3/11/25.
//

#ifndef INTENSITY_GRADIENT_RAY_H
#define INTENSITY_GRADIENT_RAY_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>
#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class ContributionRay;
}

namespace VkRender {
    class IntensityGradientRay : ScriptableEntity {
        TransformComponent *transform = nullptr;

        ContributionRay *contributionRay = nullptr;
        glm::vec3 directionCamera = glm::vec3(0.0f);
        glm::vec2 uv0 = glm::vec2(0.0f);

    public:
        void onUpdate(Timestep ts) override {}

        void onDestroy() override {}

        void onCreate() override {}
    };
}


#endif //INTENSITY_GRADIENT_RAY_H
