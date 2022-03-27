//
// Created by magnus on 3/10/22.
//

#include "Quad.h"

void Quad::setup() {
    /**
     * Create and load Mesh elements
     */
    // Prepare a model for drawing a texture onto
    model = new CRLCameraModels::Model(renderUtils.device);
    // Don't draw it before we create the texture in update()
    model->draw = false;
}


void Quad::update() {
    if (camera == nullptr)
        return;
    // If mode changes

    if (camera->modeChange) {
        auto imgConf = camera->getImageConfig();
        CRLCameraDataType textureType;
        auto lastEnabledSrc = camera->enabledSources[camera->enabledSources.size() - 1];
        switch (lastEnabledSrc) {
            case crl::multisense::Source_Chroma_Rectified_Aux:
                textureType = CrlColorImageYUV420;
                break;
            default:
                textureType = CrlGrayscaleImage;
                break;
        }
        model->prepareTextureImage(imgConf.width(), imgConf.height(), textureType);
        auto *imgData = new ImageData(((float) imgConf.width() / (float) imgConf.height()), 1);


        // Load shaders
        VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/quad.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/quad.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
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


    int runTimeInMS = (int) (renderData.runTime * 1000);
    if ((runTimeInMS % 1) < 1 && camera->play) {

        for (auto &src: camera->enabledSources) {
            switch (src) {
                case crl::multisense::Source_Chroma_Rectified_Aux:
                    model->setVideoTexture(&camera->getImage()[src],
                                           &camera->getImage()[crl::multisense::Source_Luma_Rectified_Aux]);
                    break;

                case crl::multisense::Source_Disparity_Left:
                    model->setVideoTexture(&camera->getImage()[src]);
                    break;
                default:
                    model->setVideoTexture(&camera->getImage()[src]);
                    break;
            }
        }


        count += 1;
        if (count >= 100)
            count = 1;

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


void Quad::onUIUpdate(UISettings *uiSettings) {
    camera = (CRLPhysicalCamera *) uiSettings->sharedData;
}


void Quad::draw(VkCommandBuffer commandBuffer, uint32_t i) {
    if (model->draw)
        CRLCameraModels::draw(commandBuffer, i, model);
}
