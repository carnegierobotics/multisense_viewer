//
// Created by magnus on 10/21/24.
//

#include "Viewer/VkRender/RenderResources/SyclGaussianGFX.h"

namespace VkRender{

    void SyclGaussianGFX::render(std::shared_ptr<Scene>& scene) {
        auto& registry = scene->getRegistry();

        // Find all entities with GaussianComponent
        registry.view<GaussianComponent>().each([&](auto entity, GaussianComponent& gaussianComp) {
            // Prepare output image (e.g., get from scene or create new)



        });
    }

    void SyclGaussianGFX::rasterizeGaussians(const GaussianComponent &gaussianComp) {

    }
}