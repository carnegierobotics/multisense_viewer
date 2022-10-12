//
// Created by magnus on 10/12/22.
//

#include "TrackingTest.h"

void TrackingTest::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()
    model = std::make_unique<CRLCameraModels::Model>(&renderUtils);
    model->draw = false;

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());

    vo = std::make_unique<VisualOdometry>();
}

void TrackingTest::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw)
        CRLCameraModels::draw(commandBuffer, i, model.get(), b);
}

void TrackingTest::update() {
    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::scale(mat.model, glm::vec3(0.001f, 0.001f, 0.001f));

    mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;

    auto tex = VkRender::TextureData(AR_GRAYSCALE_IMAGE, width, height);
    tex.data = (uint8_t *) malloc(width * height);
    //if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Left", &tex, 0))
    //    vo->update(tex, tex, tex);
    free(tex.data);

}


void TrackingTest::onUIUpdate(const MultiSense::GuiObjectHandles *uiHandle) {
    for (const auto &d: *uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

        previewTab = d.selectedPreviewTab;

    }
}