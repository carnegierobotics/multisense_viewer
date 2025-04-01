//
// Created by magnus on 3/11/25.
//

#ifndef ContributionRAY_H
#define ContributionRAY_H

#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class Emitter;

    class ContributionRay : ScriptableEntity {
        int quadricIndex = 0;
        Emitter* emitter = nullptr;


    public:
        glm::vec2 uv0 = glm::vec2(0.0f);

        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //ContributionRAY_H
