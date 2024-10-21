//
// Created by magnus on 10/21/24.
//

#ifndef MULTISENSE_VIEWER_SYCLGAUSSIANGFX_H
#define MULTISENSE_VIEWER_SYCLGAUSSIANGFX_H

#include <sycl/sycl.hpp>

#include "Viewer/VkRender/Components/GaussianComponent.h"
#include "Viewer/Scenes/Scene.h"

namespace VkRender {
    class SyclGaussianGFX {
    public:
        explicit SyclGaussianGFX(sycl::queue& queue)
        : m_queue(queue) {}

        void render(std::shared_ptr<Scene>& scene);

    private:
        sycl::queue& m_queue;

        void rasterizeGaussians(const GaussianComponent& gaussianComp);

    };

}


#endif //MULTISENSE_VIEWER_SYCLGAUSSIANGFX_H
