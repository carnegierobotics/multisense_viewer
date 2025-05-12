//
// Created by magnus on 3/11/25.
//

#ifndef EMITTER_H
#define EMITTER_H

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>
#include <Viewer/Rendering/PathTracer/PathTracerSYCL.h>

#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class Emitter : ScriptableEntity {
    public:
        int quadricIndex = 0;
        glm::vec3 hitPosition = glm::vec3(0.0f);
        glm::vec3 hitNormal = glm::vec3(0.0f);
        std::unique_ptr<PathTracer::PathTracerSYCL> m_pathTracerSYCL;

        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //EMITTER_H
