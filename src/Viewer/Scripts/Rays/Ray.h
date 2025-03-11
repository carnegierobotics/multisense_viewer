//
// Created by magnus on 3/11/25.
//

#ifndef RAY_H
#define RAY_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>
#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class Ray : ScriptableEntity {
        TransformComponent *transform = nullptr;

    public:
        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //RAY_H
