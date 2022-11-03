//
// Created by magnus on 10/12/22.
//

#include "TrackingTest.h"

void TrackingTest::setup() {
    // Prepare a model for drawing a texture onto
    // Don't draw it before we create the texture in update()

    Log::Logger::getInstance()->info("Setup run for {}", renderData.scriptName.c_str());

    vo = std::make_unique<VisualOdometry>();

   // m_Model.loadFromFile(Utils::getAssetsPath() + "Models/camera.gltf", renderUtils.device,renderUtils.device->m_TransferQueue, 1.0f);


    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("myScene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("myScene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};


    // Obligatory call to prepare render resources for glTFModel.
    //glTFModel::createRenderPipeline(renderUtils, shaders);

}

void TrackingTest::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (previewTab == TAB_3D_POINT_CLOUD && b);
        //glTFModel::draw(commandBuffer, i);
}

void TrackingTest::update() {
    vo->setPMat(renderData.crlCamera->get()->getCameraInfo(0).calibration, renderData.crlCamera->get()->getCameraInfo(0).imgConf.tx());

    auto left = VkRender::TextureData(AR_GRAYSCALE_IMAGE, width, height);
    left.data = (uint8_t *) malloc(width * height);
    auto right = VkRender::TextureData(AR_GRAYSCALE_IMAGE, width, height);
    right.data = (uint8_t *) malloc(width * height);
    auto depth = VkRender::TextureData(AR_DISPARITY_IMAGE, width, height);
    depth.data = (uint8_t *) malloc(width * height * 2);
    glm::vec3 translation(0.0f, 0.0f, 0.0f);
    if (renderData.crlCamera->get()->getCameraStream("Luma Rectified Left", &left, 0) &&
        renderData.crlCamera->get()->getCameraStream("Luma Rectified Right", &right, 0) &&
            renderData.crlCamera->get()->getCameraStream("Disparity Left", &depth, 0))
        translation = vo->update(left, right, depth);

    free(left.data);
    free(right.data);
    free(depth.data);

    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, (translation * glm::vec3(0.01f, 0.01f, -0.01f)));
    mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(0.001f, 0.001f, 0.001f));

    auto &d = bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &d2 = bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->m_ViewPos;
}


void TrackingTest::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

        previewTab = d.selectedPreviewTab;

    }
}