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
            auto &transform = entity.getComponent<TransformComponent>();
            transform.setScale({0.25f, 0.25f, 0.25f});
        }

        {

            auto cameraEntity = createNewCamera("DefaultCamera", 1280, 720);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
            cameraComponent.camera.setType(Camera::flycam);
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setPosition({0.0f, 0.0f, 2.5f});
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraEntity.addComponent<MeshComponent>();
        }
        {

            auto cameraEntity = createNewCamera("SecondaryCamera", 1280, 720);
            auto &cameraComponent = cameraEntity.getComponent<CameraComponent>();
            cameraComponent.camera.setType(Camera::flycam);
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setPosition({0.3, 0.0f, 2.5f});
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraEntity.addComponent<MeshComponent>();
        }


        //auto gaussianEntity = createEntity("GaussianEntity");
        //auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(Utils::getModelsPath() / "3dgs" / "3dgs_insect.ply");
        //auto &gaussianEntityModel = gaussianEntity.addComponent<OBJModelComponent>(Utils::getModelsPath() / "obj" / "quad.obj");

    }


    void MultiSenseViewer::update(uint32_t i) {

    }


}

