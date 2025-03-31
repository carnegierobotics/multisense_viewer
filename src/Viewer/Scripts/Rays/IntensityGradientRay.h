//
// Created by magnus on 3/11/25.
//

#ifndef INTENSITY_GRADIENT_RAY_H
#define INTENSITY_GRADIENT_RAY_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>
#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class IntensityGradientRay : ScriptableEntity {
        TransformComponent *transform = nullptr;

    public:
        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //INTENSITY_GRADIENT_RAY_H
