//
// Created by magnus on 5/8/22.
//

#include "DisparityPreview.h"
#include "GLFW/glfw3.h"


void DisparityPreview::setup(Base::Render r) {
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, AR_CAMERA_DATA_COLOR_IMAGE);

    // Don't draw it before we create the texture in update()
    model->draw = false;

    for (auto dev : *r.gui){
        if (dev.streams.find(AR_PREVIEW_DISPARITY) == dev.streams.end()  || dev.state != AR_STATE_ACTIVE)
            continue;

        auto opt = dev.streams.find(AR_PREVIEW_DISPARITY)->second;
        r.crlCamera->get()->camPtr->start(opt.selectedStreamingMode, opt.selectedStreamingSource);
    }

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());

}


void DisparityPreview::update(CameraConnection *conn) {
    if (playbackSate != AR_PREVIEW_PLAYING)
        return;

    auto *camera = conn->camPtr;
    assert(camera != nullptr);


    if (!model->draw) {
        auto imgConf = camera->getCameraInfo().imgConf;

        std::string vertexShaderFileName;
        std::string fragmentShaderFileName;
        vertexShaderFileName = "myScene/spv/depth.vert";
        fragmentShaderFileName = "myScene/spv/depth.frag";

        model->prepareTextureImage(imgConf.width(), imgConf.height(), AR_DISPARITY_IMAGE);

        auto *imgData = new ImageData(((float) imgConf.width() / (float) imgConf.height()), 1);


        // Load shaders
        VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        // Create a quad and store it locally on the GPU
        model->createMeshDeviceLocal((ArEngine::Vertex *) imgData->quad.vertices,
                                     imgData->quad.vertexCount, imgData->quad.indices, imgData->quad.indexCount);


        // Create graphics render pipeline
        CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

        model->draw = true;
    }

    if (model->draw) {
        crl::multisense::image::Header *disp;
        camera->getCameraStream(src, disp);

        model->setGrayscaleTexture(disp);

    }


    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);

    mat.model = glm::translate(mat.model, glm::vec3(2.3, up, -5));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    auto *d = (UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;
}


void DisparityPreview::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    posX = uiHandle.sliderOne;
    posY = uiHandle.sliderTwo;
    posZ = uiHandle.sliderThree;
    if (uiHandle.keypress == GLFW_KEY_I){
        up += 0.1;
    }
    if (uiHandle.keypress == GLFW_KEY_K){
        up -= 0.1;
    }

    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        if (dev.streams.find(AR_PREVIEW_DISPARITY) == dev.streams.end() || dev.state != AR_STATE_ACTIVE)
            continue;

        src = dev.streams.find(AR_PREVIEW_DISPARITY)->second.selectedStreamingSource;
        playbackSate = dev.streams.find(AR_PREVIEW_DISPARITY)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;

    }
}


void DisparityPreview::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_2D_PREVIEW)
        CRLCameraModels::draw(commandBuffer, i, model);
}
