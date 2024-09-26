//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/Application/Application.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {

    MultiSenseViewer::MultiSenseViewer(Application &ctx, const std::string& name) : Scene(name) {
        {
            auto entity = createEntity("S30Camera");
            auto &modelComponent = entity.addComponent<MeshComponent>(Utils::getModelsPath() / "obj" / "s30.obj");
            //auto &modelComponent = entity.addComponent<MeshComponent>("/home/magnus/phd/SuGaR/output/refined_mesh/0005/sugarfine_3Dgs30000_sdfestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj");
            auto &transform = entity.getComponent<TransformComponent>();
            //transform.setScale({0.25f, 0.25f, 0.25f});
        }
        {
            auto cameraEntity = createNewCamera("DefaultCamera", 400, 400);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
            cameraComponent.camera.setType(Camera::flycam);
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setPosition({0.0f, 0.0f, 2.5f});
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraEntity.addComponent<MeshComponent>(1);
        }

#ifdef SYCL_ENABLED
        {  auto gaussianEntity = createEntity("GaussianEntity");
           auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(Utils::getModelsPath() / "3dgs" / "3dgs_insect.ply");

            int debug = 1;
        }
#endif

    }


    void MultiSenseViewer::update(uint32_t i) {

    }


}

