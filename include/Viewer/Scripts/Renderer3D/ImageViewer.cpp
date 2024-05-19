//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/ImageViewer.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Renderer/Components/CustomModels.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"

void ImageViewer::setup() {

    auto quad = m_context->createEntity("3dgs_image");
    auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(
            Utils::getModelsPath() / "obj" / "quad.obj",
            m_context->renderUtils.device);

    quad.addComponent<VkRender::SecondaryRenderViewComponent>();

    auto &res = quad.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils,
                                                                               "SYCLRenderer.vert.spv",
                                                                               "SYCLRenderer.frag.spv");
    res.bind(modelComponent);

    m_syclSobel = std::make_unique<GaussianRenderer>(m_context->getCamera());

    Widgets::make()->button(WIDGET_PLACEMENT_RENDERER3D, "btn", &btn);

}


void ImageViewer::update() {
    auto &camera = m_context->getCamera();
    if (btn){
        std::cout << "View Matrix:\n";
        std::cout << glm::to_string(camera.matrices.view) << "\n";
    }
    auto imageView = m_context->findEntityByName("3dgs_image");
    if (imageView) {
        auto &obj = imageView.getComponent<VkRender::DefaultGraphicsPipelineComponent2>();
        obj.mvp.projection = camera.matrices.perspective;
        obj.mvp.view = camera.matrices.view;
        auto model = glm::mat4(1.0f);
        float xOffsetPx = (m_context->renderData.width - 150.0) / m_context->renderData.width;
        float translationX = xOffsetPx * 2 - 1;
        float translationY = xOffsetPx * 2 - 1;
        model = glm::translate(model, glm::vec3(translationX, translationY, 0.0f));
        float scaleX = 300.0f / m_context->renderData.width;
        model = glm::scale(model, glm::vec3(scaleX, scaleX, 1.0f)); // Uniform scaling in x and y
        obj.mvp.model = glm::mat4(0.2f);
    }
}

