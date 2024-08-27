//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {

    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) {
        m_sceneName = "MultiSense Viewer";

        {
            auto entity = createEntity("S30Camera");
            auto &modelComponent = entity.addComponent<MeshComponent>(Utils::getModelsPath() / "obj" / "s30.obj");
            //auto &modelComponent = entity.addComponent<MeshComponent>("/home/magnus/phd/SuGaR/output/refined_mesh/0017/3dgs.obj");
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
        {
            auto gaussianEntity = createEntity("GaussianEntity");
            auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(
                    Utils::getModelsPath() / "3dgs" / "3dgs.ply");
            int debug = 1;
        }
#endif

    }


    void MultiSenseViewer::update(uint32_t i) {

    }


}

