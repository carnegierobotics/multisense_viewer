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

        auto entity = createEntity("FirstEntity");
        auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "s30.obj");

        auto e = createEntity("DefaultCamera");
        auto &c = e.addComponent<CameraComponent>(Camera(1280, 720));
        c.camera.setType(Camera::flycam);
        auto &transform = e.getComponent<TransformComponent>();
        transform.setPosition({0.0f, 0.0f, 4.0f});
        c.camera.pose.pos = transform.getPosition();
        auto &gizmo = e.addComponent<CameraModelComponent>();


        auto gaussianEntity = createEntity("GaussianEntity");
        auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(
                Utils::getModelsPath() / "3dgs" / "3dgs.ply");
        auto &gaussianEntityModel = gaussianEntity.addComponent<OBJModelComponent>(Utils::getModelsPath() / "obj" / "quad.obj");

    }


    void MultiSenseViewer::update(uint32_t i) {

    }


}

