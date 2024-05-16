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

    m_syclRenderer = std::make_unique<SYCLRayTracer>(m_context->renderData.width, m_context->renderData.height);
    m_syclRenderer->save_image("../sycl.ppm",  m_context->renderData.width,  m_context->renderData.height);
    auto syclOutput = m_syclRenderer->get_image_8bit(m_context->renderData.width,  m_context->renderData.height);

    uint32_t size = syclOutput.size() * sizeof(SYCLRayTracer::Pixel);

    syclRenderTarget.fromBuffer(syclOutput.data(), size, VK_FORMAT_R8G8B8A8_UNORM, m_context->renderData.width, m_context->renderData.height, m_context->renderUtils.device, m_context->renderUtils.device->m_TransferQueue);
    //syclRenderTarget.fromKtxFile((Utils::getTexturePath() / "empty.ktx").string(), VK_FORMAT_R8G8B8A8_UNORM, m_context->renderUtils.device, m_context->renderUtils.device->m_TransferQueue);

    res.setTexture(&syclRenderTarget.m_descriptor);
}


void ImageViewer::update() {
    auto &camera = m_context->getCamera();

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

