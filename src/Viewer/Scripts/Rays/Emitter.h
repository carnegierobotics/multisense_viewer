//
// Created by magnus on 3/11/25.
//

#ifndef EMITTER_H
#define EMITTER_H


#include "Viewer/Scenes/ScriptableEntity.h"

//#include <Viewer/Rendering/PathTracer/PathTracerSetup.h>

namespace VkRender {
    class Emitter2 : ScriptableEntity {
    public:
        int quadricIndex = 0;
        glm::vec3 hitPosition = glm::vec3(0.0f);
        glm::vec3 hitNormal = glm::vec3(0.0f);
        std::unique_ptr<PathTracer::PathTracerSetup> m_PathTracerSetup;

        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //EMITTER_H
