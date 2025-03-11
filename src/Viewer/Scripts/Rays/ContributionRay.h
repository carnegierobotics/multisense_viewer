//
// Created by magnus on 3/11/25.
//

#ifndef ContributionRAY_H
#define ContributionRAY_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>

#include <yaml-cpp/emitter.h>

#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class Emitter;

    class ContributionRay : ScriptableEntity {

      int quadricIndex = 0;

        Emitter* emitter = nullptr;

    public:
        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //ContributionRAY_H
