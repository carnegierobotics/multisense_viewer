//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"
#include "Viewer/VkRender/Components/CameraModelComponent.h"

namespace VkRender {

    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) {
        m_sceneName = "MultiSense Viewer";

        {
            auto entity = createEntity("FirstEntity");
            auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
                    Utils::getModelsPath() / "obj" / "s30.obj");
            auto &transform = entity.getComponent<TransformComponent>();
            transform.setScale({0.25f, 0.25f, 0.25f});
        }

        {
            auto cameraEntity = createEntity("DefaultCamera");
            auto &cameraComponent = cameraEntity.addComponent<CameraComponent>(Camera(1280, 720));
            cameraComponent.camera.setType(Camera::flycam);
            auto &transform = cameraEntity.getComponent<TransformComponent>();
            transform.setPosition({0.0f, 0.0f, 2.5f});
            cameraComponent.camera.pose.pos = transform.getPosition();
            cameraEntity.addComponent<CameraModelComponent>();
        }


        //auto gaussianEntity = createEntity("GaussianEntity");
        //auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(Utils::getModelsPath() / "3dgs" / "3dgs_insect.ply");
        //auto &gaussianEntityModel = gaussianEntity.addComponent<OBJModelComponent>(Utils::getModelsPath() / "obj" / "quad.obj");

    }


    void MultiSenseViewer::update(uint32_t i) {

    }


}

