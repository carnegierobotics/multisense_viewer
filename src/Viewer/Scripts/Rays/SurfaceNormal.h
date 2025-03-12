//
// Created by magnus on 3/11/25.
//

#ifndef SURFACE_NORMAL_RAY_H
#define SURFACE_NORMAL_RAY_H


#include "Emitter.h"
#include "Viewer/Scenes/ScriptableEntity.h"

namespace VkRender {
    class SurfaceNormal : ScriptableEntity {
        int quadricIndex = 0;
        Emitter* emitter = nullptr;

    public:
        void onUpdate(Timestep ts) override;

        void onDestroy() override;

        void onCreate() override;
    };
}


#endif //SURFACE_NORMAL_RAY_H
