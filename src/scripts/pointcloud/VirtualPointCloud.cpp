#include <MultiSense/src/imgui/Layer.h>
#include "VirtualPointCloud.h"
#include "glm/gtx/string_cast.hpp"


void VirtualPointCloud::setup(Base::Render r) {

    model = new CRLCameraModels::Model(nullptr);
    model->setTexture(Utils::getTexturePath() + "neist_point.jpg");


    //r.crlCamera->preparePointCloud(960, 600);
    model->createEmtpyTexture(960, 600, AR_POINT_CLOUD);

    VkPipelineShaderStageCreateInfo vs = loadShader("myScene/spv/pointcloud.vert", VK_SHADER_STAGE_VERTEX_BIT);
    VkPipelineShaderStageCreateInfo fs = loadShader("myScene/spv/pointcloud.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{vs},
                                                            {fs}};

    CRLCameraModels::createRenderPipeline(shaders, model, type, &renderUtils);

    auto *buf = (ArEngine::PointCloudParam *) bufferThreeData;
    //buf->kInverse = r.crlCamera->getCameraInfo().kInverseMatrix;
    buf->height = static_cast<float>(960.0f);
    buf->width = static_cast<float>(600.0f);
    std::cout << glm::to_string(buf->kInverse) << std::endl;

    meshData = new ArEngine::Vertex[vertexCount]; // Don't forget to delete [] a; when you're done!

    int v = 0;
    for (int i = 0; i < 960; ++i) {
        for (int j = 0; j < 600; ++j) {
            meshData[v].pos = glm::vec3((float) i / 100.0f, (float) j / 100.0f, 0.0f);
            meshData[v].uv0 = glm::vec2((float) 1 - ((float) i / 960.0f), (float) 1 - ((float) j / 600.0f));
            v++;
        }
    }


}


void VirtualPointCloud::update() {
    if (playbackSate != AR_PREVIEW_PLAYING && TAB_3D_POINT_CLOUD == selectedPreviewTab)
        return;
    //CRLBaseInterface *camPtr = conn->camPtr;


    crl::multisense::image::Header disp;
    //camPtr->getCameraStream(nullptr);
    //model->setTexture(&disp);

    free((void *) disp.imageDataP);

    model->createMesh((ArEngine::Vertex *) meshData, vertexCount);


    ArEngine::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, -5.0f));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    //mat.model = glm::translate(mat.model, glm::vec3(2.8, 0.4, -5));
    auto *d = (ArEngine::UBOMatrix *) bufferOneData;
    d->model = mat.model;
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto *d2 = (ArEngine::FragShaderParams *) bufferTwoData;
    d2->objectColor = glm::vec4(0.25f, 0.25f, 0.25f, 1.0f);
    d2->lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    d2->lightPos = glm::vec4(glm::vec3(0.0f, -3.0f, 0.0f), 1.0f);
    d2->viewPos = renderData.camera->viewPos;
}


void VirtualPointCloud::onUIUpdate(AR::GuiObjectHandles uiHandle) {
    // GUi elements if a PHYSICAL camera has been initialized
    for (const auto &dev: *uiHandle.devices) {
        if (dev.button)
            model->draw = false;

        /*
        if (dev.streams.find(AR_PREVIEW_VIRTUAL_POINT_CLOUD) == dev.streams.end())
            continue;

        playbackSate = dev.streams.find(AR_PREVIEW_VIRTUAL_POINT_CLOUD)->second.playbackStatus;
        selectedPreviewTab = dev.selectedPreviewTab;
    */
    }

}


void VirtualPointCloud::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (model->draw && playbackSate != AR_PREVIEW_NONE && selectedPreviewTab == TAB_3D_POINT_CLOUD)
        CRLCameraModels::draw(commandBuffer, i, model, false);
}