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

    /*
    std::filesystem::path filePath = "/home/magnus/crl/multisense_viewer/3dgs_insect.ply";
    Log::Logger::getInstance()->info("Loading new model from {}", filePath.string());
    entityName = filePath.string();
    auto entity = m_context->createEntity(entityName);
    auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
            Utils::getModelsPath() / "obj" / "quad.obj",
            m_context->renderUtils.device);
    entity.addComponent<VkRender::SecondaryRenderViewComponent>();
    auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils,
                                                                                 "SYCLRenderer.vert.spv",
                                                                                 "SYCLRenderer.frag.spv");
    res.bind(modelComponent);
    const auto &camera = m_context->getCamera();
    m_gaussianRenderer = std::make_unique<GaussianRenderer>(camera);

    m_syclRenderTarget = std::make_unique<TextureVideo>(camera.m_width, camera.m_height,
                                                        m_context->renderUtils.device,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VK_FORMAT_R8G8B8A8_UNORM);
    res.setTexture(&m_syclRenderTarget->m_descriptor);
    //m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(filePath, 1);

   // m_gaussianRenderer->setupBuffers(m_context->getCamera());
    Widgets::make()->button(WIDGET_PLACEMENT_RENDERER3D, "RunCPU", &btn);

     */
}

void ImageViewer::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
}


void ImageViewer::update() {
    auto &camera = m_context->getCamera();
    if (btn) {
        /*
        if (!m_gaussianRenderer->loadedPly) {
            m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(
                    "/home/magnus/crl/multisense_viewer/3dgs_insect.ply", 1);
            m_gaussianRenderer->loadedPly = true;
        } else {
            m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(
                    "/home/magnus/crl/multisense_viewer/3dgs_coordinates.ply", 1);
            m_gaussianRenderer->loadedPly = false;
        }
        m_gaussianRenderer->setupBuffers(m_context->getCamera());
        */

    }

    auto imageView = m_context->findEntityByName(entityName);
    if (imageView) {

        //m_gaussianRenderer->tileRasterizer(camera, btn);
        auto *dataPtr = m_syclRenderTarget->m_DataPtr;
        uint32_t size = camera.m_height * camera.m_width * 4;
        //std::memcpy(dataPtr, m_gaussianRenderer->img, size);
        m_syclRenderTarget->updateTextureFromBuffer();

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

void ImageViewer::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    if (uiHandle->m_paths.update3DGSPath) {
        /*
        if (m_context->findEntityByName(entityName))
            return;
        Log::Logger::getInstance()->info("Loading new model from {}", uiHandle->m_paths.importFilePath.string());
        entityName = uiHandle->m_paths.importFilePath.filename().string();
        auto entity = m_context->createEntity(entityName);
        auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "quad.obj",
                m_context->renderUtils.device);
        entity.addComponent<VkRender::SecondaryRenderViewComponent>();
        auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils,
                                                                                      "SYCLRenderer.vert.spv",
                                                                                      "SYCLRenderer.frag.spv");
        res.bind(modelComponent);
        const auto &camera = m_context->getCamera();
        m_gaussianRenderer = std::make_unique<GaussianRenderer>(camera);

        m_syclRenderTarget = std::make_unique<TextureVideo>(camera.m_width, camera.m_height,
                                                            m_context->renderUtils.device,
                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                            VK_FORMAT_R8G8B8A8_UNORM);
        res.setTexture(&m_syclRenderTarget->m_descriptor);
        m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(
                uiHandle->m_paths.importFilePath.string(), 1);
        m_gaussianRenderer->setupBuffers(m_context->getCamera());

        */
    }

}

