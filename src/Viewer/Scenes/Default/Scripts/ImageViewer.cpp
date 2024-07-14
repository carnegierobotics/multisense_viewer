//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scenes/Renderer3D/ImageViewer.h"
#include "Viewer/VkRender/ImGui/Widgets.h"
#include "Viewer/VkRender/Components.h"
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"
#include "Viewer/VkRender/Components/RenderComponents/DefaultGraphicsPipelineComponent2.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"


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
    auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->renderUtils,
                                                                                 "SYCLRenderer.vert.spv",
                                                                                 "SYCLRenderer.frag.spv");
    res.bind(modelComponent);
    res.setTexture(&m_syclRenderTarget->m_descriptor);

    VkRender::InitializeInfo initInfo{};
    initInfo.height = camera.m_height;
    initInfo.width = camera.m_width;
    initInfo.channels = 4;
    initInfo.camera = &camera;
    initInfo.context = m_context;
    initInfo.imageSize = camera.m_height * camera.m_width * 4; // RGBA-uint

    std::filesystem::path filePath = "/home/magnus/crl/multisense_viewer/3dgs_insect.ply";
    //m_gaussianRenderer->gs = GaussianRenderer::loadFromFile(filePath, 1);
    // m_gaussianRenderer->setupBuffers(m_context->getCamera());
    m_renderer = std::make_unique<VkRender::GaussianRenderer>();

    splatEntity = "Default 3DGS model";
    m_context->createEntity(splatEntity);

    m_renderer->setup(initInfo, useCPU);
    prevDevice = useCPU;

    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Use CPU", &useCPU);
    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "3DGS Render", &render3dgs);
    Widgets::make()->button(WIDGET_PLACEMENT_RENDERER3D, "3DGS Single image", &render3dgsImage);
}

void ImageViewer::onWindowResize(const VkRender::GuiObjectHandles *uiHandle) {
    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "Use CPU", &useCPU);
    Widgets::make()->checkbox(WIDGET_PLACEMENT_RENDERER3D, "3DGS Render", &render3dgs);
    Widgets::make()->button(WIDGET_PLACEMENT_RENDERER3D, "3DGS Single image", &render3dgsImage);
}


void ImageViewer::update() {

    if (useCPU != prevDevice) {
        const auto &camera = m_context->getCamera();
        VkRender::InitializeInfo initInfo{};
        initInfo.height = camera.m_height;
        initInfo.width = camera.m_width;
        initInfo.channels = 4;
        initInfo.camera = &camera;
        initInfo.context = m_context;
        initInfo.imageSize = camera.m_height * camera.m_width * 4; // RGBA-uint

        m_renderer = std::make_unique<VkRender::GaussianRenderer>();
        splatEntity = "Default 3DGS model";
        m_context->createEntity(splatEntity);
        m_renderer->setup(initInfo, useCPU);
    }
    prevDevice = useCPU;

    auto &camera = m_context->getCamera();
    VkRender::RenderInfo info{};
    info.camera = &camera;
    info.debug = btn;

    if (btn) {
        m_renderer->singleOneSweep();
    }

    if (render3dgsImage || render3dgs && m_context->findEntityByName(splatEntity)) {
        auto startRender = std::chrono::high_resolution_clock::now();

        m_renderer->render(info, &m_context->renderUtils);

        auto endRender = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationRender = endRender - startRender;
        auto startUpdateTexture = std::chrono::high_resolution_clock::now();

        if (m_renderer->getImage())
            m_syclRenderTarget->updateTextureFromBuffer(m_renderer->getImage(), m_renderer->getImageSize());

        auto endUpdateTexture = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationUpdateTexture = endUpdateTexture - startUpdateTexture;

        Log::Logger::getInstance()->trace("Full Render: {}ms, update  tex{}us",
                                          std::chrono::duration_cast<std::chrono::milliseconds>(durationRender).count(),
                                          std::chrono::duration_cast<std::chrono::microseconds>(
                                                  durationUpdateTexture).count());
    }

}

void ImageViewer::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {

    if (uiHandle->m_paths.update3DGSPath) {
        m_renderer->gs = VkRender::GaussianRenderer::loadFromFile(uiHandle->m_paths.importFilePath.string(), 1);
        splatEntity = uiHandle->m_paths.importFilePath.filename();
        m_renderer->setupBuffers(&m_context->getCamera());
        m_context->createEntity(splatEntity);

    }

}

