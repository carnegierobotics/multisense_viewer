//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/DefaultScene.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Renderer/Components/CustomModels.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"

void DefaultScene::setup() {

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Grid", &enable);

    auto grid = m_context->createEntity("3DViewerGrid");
    grid.addComponent<VkRender::CustomModelComponent>(&m_context->renderUtils);
    auto uuid = grid.getUUID();

    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

    /*
{
    auto entity = m_context->createEntity("viking_room");
    auto &component = entity.addComponent<VkRender::OBJModelComponent>(
            Utils::getModelsPath() / "obj" / "viking_room.obj", m_context->renderUtils.device);
    entity.addComponent<VkRender::SecondaryRenderPassComponent>();

    entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(
            &m_context->renderUtils).bind(component);

}

*/

    auto cameraGizmo = m_context->findEntityByName("Default");
    auto &rr = cameraGizmo.addComponent<VkRender::CameraGraphicsPipelineComponent>(&m_context->renderUtils);


    if (showDepthView) {
        auto quad = m_context->createEntity("depthImageView");
        auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "quad.obj",
                m_context->renderUtils.device);

        quad.addComponent<VkRender::ImageViewComponent>();

        auto &res = quad.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils,
                                                                                   "default2D.vert.spv",
                                                                                   "default2D.frag.spv");
        res.bind(modelComponent);
        res.setTexture(&m_context->renderUtils.depthRenderPass->depthImageInfo);
    }

}


void DefaultScene::update() {
    auto &camera = m_context->getCamera();

    auto e = m_context->findEntityByName("3DViewerGrid");
    if (e) {
        auto &c = e.getComponent<VkRender::CustomModelComponent>();
        c.mvp.model = glm::mat4(1.0f);
        c.mvp.view = camera.matrices.view;
        c.mvp.projection = camera.matrices.perspective;
        c.model->draw = enable;
        c.update(m_context->renderUtils.swapchainIndex);

    }

    auto cameraEntity = m_context->findEntityByName("Default");
    if (cameraEntity && cameraEntity.hasComponent<VkRender::CameraComponent>()){
        auto& camModel = cameraEntity.getComponent<VkRender::CameraComponent>().camera;
        auto& cameraGizmo = cameraEntity.getComponent<VkRender::CameraGraphicsPipelineComponent>();
        cameraGizmo.mvp.model = glm::scale(glm::inverse(camModel.matrices.view), glm::vec3(0.2f, 0.2f, 0.2f));
        cameraGizmo.mvp.view = camera.matrices.view;
        cameraGizmo.mvp.projection = camera.matrices.perspective;
    }

    if (showDepthView) {
        auto depthImageView = m_context->findEntityByName("depthImageView");
        if (depthImageView) {
            auto &obj = depthImageView.getComponent<VkRender::DefaultGraphicsPipelineComponent2>();
            obj.mvp.projection = camera.matrices.perspective;
            obj.mvp.view = camera.matrices.view;

            auto model = glm::mat4(1.0f);
            float xOffsetPx = (m_context->renderData.width - 150.0) / m_context->renderData.width;
            float translationX = xOffsetPx * 2 - 1;
            float translationY = xOffsetPx * 2 - 1;
            model = glm::translate(model, glm::vec3(translationX, translationY, 0.0f));
            float scaleX = 300.0f / m_context->renderData.width;
            model = glm::scale(model, glm::vec3(scaleX, scaleX, 1.0f)); // Uniform scaling in x and y

            obj.mvp.model = model;
        }
    }


    auto gsMesh = m_context->findEntityByName("viking_room");
    if (gsMesh) {
        auto &obj = gsMesh.getComponent<VkRender::DefaultGraphicsPipelineComponent2>();
        obj.mvp.projection = camera.matrices.perspective;
        obj.mvp.view = camera.matrices.view;
        obj.mvp.model = glm::mat4(1.0f);
    }

}


void DefaultScene::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    auto &currentCamera = m_context->getCamera();

    // Loop through cameras and update the camera gizmo position
    for (auto entity: m_context->m_registry.view<VkRender::CameraGraphicsPipelineComponent>()) {
        /*
        auto &obj = m_context->m_registry.get<VkRender::CameraGraphicsPipelineComponent>(entity);
        auto &camComponent = m_context->m_registry.get<VkRender::CameraComponent>(entity);
        obj.mvp.projection = currentCamera.matrices.perspective;
        obj.mvp.view = currentCamera.matrices.view;

        glm::mat4 model = glm::translate(glm::mat4(1.0f), camComponent.camera.pose.pos);
        model = model * glm::mat4_cast(camComponent.camera.pose.q);
        model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
        obj.mvp.model = model;
        obj.mvp.camPos = camComponent.camera.pose.pos;
        */
    }
}

void DefaultScene::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {

    if (showDepthView) {
        // Re-create the depth image view if we resize the viewport
        auto oldEntity = m_context->findEntityByName("depthImageView");
        if (oldEntity) {
            oldEntity.getComponent<VkRender::DefaultGraphicsPipelineComponent2>().cleanUp(0, true);
            m_context->destroyEntity(m_context->findEntityByName("depthImageView"));
        }
        auto quad = m_context->createEntity("depthImageView");
        auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "quad.obj",
                m_context->renderUtils.device);


        auto &res = quad.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils,
                                                                                   "default2D.vert.spv",
                                                                                   "default2D.frag.spv");
        quad.addComponent<VkRender::ImageViewComponent>();

        res.bind(modelComponent);
        res.setTexture(&m_context->renderUtils.depthRenderPass->depthImageInfo);
    }

}
