//
// Created by magnus on 3/10/22.
//

#include "DefaultPreview.h"

void DefaultPreview::setup() {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device, CrlImage);
    // Don't draw it before we create the texture in update()
    model->draw = false;
}


void DefaultPreview::update(CameraConnection* conn) {
    auto* camera = conn->camPtr;
    assert(camera != nullptr);

    if (camera->modeChange) {
        auto imgConf = camera->getImageConfig();
        CRLCameraDataType textureType;
        auto lastEnabledSrc = camera->enabledSources[camera->enabledSources.size() - 1];


        std::string vertexShaderFileName;
        std::string fragmentShaderFileName;

        switch (lastEnabledSrc) {
            case crl::multisense::Source_Chroma_Rectified_Aux:
            case crl::multisense::Source_Chroma_Aux:
            case crl::multisense::Source_Chroma_Left:
                textureType = CrlColorImageYUV420;
                vertexShaderFileName = "myScene/spv/quad.vert";
                fragmentShaderFileName = "myScene/spv/quad.frag";
                break;
            case crl::multisense::Source_Disparity_Left:
                textureType = CrlDisparityImage;
                vertexShaderFileName = "myScene/spv/depth.vert";
                fragmentShaderFileName = "myScene/spv/depth.frag";
                break;
            default:
                vertexShaderFileName = "myScene/spv/depth.vert";
                fragmentShaderFileName = "myScene/spv/depth.frag";
                textureType = CrlGrayscaleImage;
                break;
        }
        model->prepareTextureImage(imgConf.width(), imgConf.height(), textureType);
        auto *imgData = new ImageData(((float) imgConf.width() / (float) imgConf.height()), 1);


        // Load shaders
        VkPipelineShaderStageCreateInfo vs = loadShader(vertexShaderFileName, VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader(fragmentShaderFileName, VK_SHADER_STAGE_FRAGMENT_BIT);
        std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                                {fs}};
        // Create quad and store it locally on the GPU
        model->createMeshDeviceLocal((CRLCameraModels::Model::Vertex *) imgData->quad.vertices,
                                     imgData->quad.vertexCount, imgData->quad.indices, imgData->quad.indexCount);

        // Create graphics render pipeline
        CRLCameraModels::createRenderPipeline(renderUtils, shaders, model, type);

        model->draw = true;
        camera->modeChange = false;
    }

    if (camera->play) {

        auto src = camera->enabledSources[camera->enabledSources.size() - 1];
        switch (src) {
            case crl::multisense::Source_Chroma_Rectified_Aux:
            case crl::multisense::Source_Chroma_Aux:
            case crl::multisense::Source_Chroma_Left:
                model->setColorTexture(&camera->getImage()[src],
                                       &camera->getImage()[crl::multisense::Source_Luma_Rectified_Aux]);
                break;

            default:
                model->setGrayscaleTexture(&camera->getImage()[src]);
                break;
        }
    }



    UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(5.0f, -5.0f, -5.0f));
    mat.model = glm::rotate(mat.model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    mat.model = glm::scale(mat.model, glm::vec3(5.0f, 5.0f, 5.0f));

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


void DefaultPreview::onUIUpdate(GuiObjectHandles uiHandle) {
    //camera = (CRLPhysicalCamera *) uiSettings->physicalCamera;

}


void DefaultPreview::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw)
        CRLCameraModels::draw(commandBuffer, i, model);
}
