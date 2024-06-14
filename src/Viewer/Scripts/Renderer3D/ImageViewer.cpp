//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/ImageViewer.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"

#include "Viewer/SYCL/RayTracer.h"


void ImageViewer::setup() {
    const auto &camera = m_context->getCamera();
    m_syclRenderTarget = std::make_unique<TextureVideo>(camera.m_width, camera.m_height,
                                                        m_context->renderUtils.device,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VK_FORMAT_R8G8B8A8_UNORM);
    auto entity = m_context->createEntity("SecondaryView");
    entity.addComponent<VkRender::SecondaryRenderViewComponent>();
    auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
            Utils::getModelsPath() / "obj" / "quad.obj",
            m_context->renderUtils.device);
    auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils, "SYCLRenderer.vert.spv", "SYCLRenderer.frag.spv");
    res.bind(modelComponent);
    res.setTexture(&m_syclRenderTarget->m_descriptor);

    VkRender::AbstractRenderer::InitializeInfo initInfo{};
    initInfo.height = camera.m_height;
    initInfo.width = camera.m_width;
    initInfo.channels = 4;
    initInfo.camera = &camera;
    initInfo.context = m_context;
    initInfo.imageSize = camera.m_height *  camera.m_width * 4; // RGBA-uint

#ifdef SYCL_ENABLED
    std::filesystem::path filePath = "/home/magnus/crl/multisense_viewer/3dgs_insect.ply";
        //m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(filePath, 1);
   // m_gaussianRenderer->setupBuffers(m_context->getCamera());
    m_renderer = std::make_unique<VkRender::GaussianRenderer>();

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "SortGPU", &btn);
    splatEntity = "Default 3DGS model";
    m_context->createEntity(splatEntity);
    m_renderer->setup(initInfo);
#endif

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "RenderCustomView", &renderImage);
}

void ImageViewer::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "SortGPU", &btn);
    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "RenderCustomView", &renderImage);

}


void ImageViewer::update() {
    if (!renderImage)
        return;
    auto &camera = m_context->getCamera();
    VkRender::AbstractRenderer::RenderInfo info{};
    info.camera = &camera;
    info.debug =  btn;
#ifdef SYCL_ENABLED
    if (m_context->findEntityByName(splatEntity)){
        auto startRender = std::chrono::high_resolution_clock::now();

        m_renderer->render(info, &m_context->renderUtils);

        auto endRender = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationRender = endRender - startRender;
        auto startUpdateTexture = std::chrono::high_resolution_clock::now();

        m_syclRenderTarget->updateTextureFromBuffer(m_renderer->getImage(), m_renderer->getImageSize());

        auto endUpdateTexture = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationUpdateTexture = endUpdateTexture - startUpdateTexture;

        Log::Logger::getInstance()->traceWithFrequency("tag123", 100, "Render: {}ms, update {}us", std::chrono::duration_cast<std::chrono::milliseconds>(durationRender).count(), std::chrono::duration_cast<std::chrono::microseconds>(durationUpdateTexture).count());
    }
#endif

}

void ImageViewer::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    if (uiHandle->m_paths.update3DGSPath) {
#ifdef SYCL_ENABLED
        m_renderer->gs = VkRender::GaussianRenderer::loadFromFile(uiHandle->m_paths.importFilePath.string(), 1);
        splatEntity = uiHandle->m_paths.importFilePath.filename();
        m_renderer->setupBuffers(&m_context->getCamera());
        m_context->createEntity(splatEntity);
#else
        std::string filePath = uiHandle->m_paths.importFilePath.string();
        Log::Logger::getInstance()->info("Loading new 3DGS file from {}", filePath.c_str());
        //m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(uiHandle->m_paths.importFilePath.string(), 1);
        //m_gaussianRenderer->setupBuffers(m_context->getCamera());
#endif
    }

}

