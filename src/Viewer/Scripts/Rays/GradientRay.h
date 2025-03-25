//
// Created by magnus on 3/11/25.
//

#ifndef GradientRay_H
#define GradientRay_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>
#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class GradientRay : ScriptableEntity {
        TransformComponent *transform = nullptr;
    public:
        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;

        glm::vec3 ray;
        glm::vec3 origin;

    };
}


#endif //GradientRay_H
