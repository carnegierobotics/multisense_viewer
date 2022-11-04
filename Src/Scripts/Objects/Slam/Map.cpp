//
// Created by magnus on 11/3/22.
//

#include "Map.h"
#include "opencv2/opencv.hpp"
#include "MultiSense/Src/VO/LazyCSV.h"

void Map::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("myScene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("myScene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};
    float fx = 868.246;
    float fy = 868.246;
    float cx = 516.0;
    float cy = 386.0;
    m_PLeft = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
    fx = 868.246;
    fy = 868.246;
    cx = 516.0;
    cy = 386.0;
    m_PRight = (cv::Mat_<float>(3, 4) << fx, 0., cx, -78.045330571, 0., fy, cy, 0., 0, 0., 1., 0.);
    lazycsv::parser parser{ "../Slam/G0/G-0_ground_truth/gt_6DoF_gnss_and_imu.csv" };

    std::vector<std::string_view> coords;

    gtPositions.reserve(10000);
    for (const auto row : parser) {
        try {
            const auto [time, x, y, z] = row.cells(0, 1, 2, 3); // indexes must be in ascending order

            float xPos = std::stof(std::string(x.trimed()));
            float yPos = std::stof(std::string(y.trimed()));
            float zPos = std::stof(std::string(z.trimed()));
            gtPositions.push_back({xPos, yPos, zPos});
        } catch (...) {
            printf("csv read error\n");
        }
    }

    /** GT Traces models **/
    size_t objects = 293;
    m_TruthTraces.resize(objects);
    float scale = 5.0f;
    for (int i = 0; i < objects; ++i){
        m_TruthTraces[i] = std::make_unique<glTFModel::Model>(renderUtils.device);
        m_TruthTraces[i]->translate(gtPositions[150 + (i * 200)].getVec() * glm::vec3(scale, scale, scale));
        m_TruthTraces[i]->scale(glm::vec3(1.0f/scale, 1.0f/scale, 1.0f/scale));
        m_TruthTraces[i]->loadFromFile(Utils::getAssetsPath() + "Models/Box/glTF/Box.gltf", renderUtils.device, renderUtils.device->m_TransferQueue, 1.0f);
        m_TruthTraces[i]->createRenderPipeline(renderUtils, shaders);
    }
}

void Map::recv(void* data){


}

void Map::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (frame % 200 == 0){
        drawBoxes++;
    }
    if (drawBoxes >= 293)
        drawBoxes = 0;

    for(size_t j = 0; j < drawBoxes; ++j){
        m_TruthTraces[j]->draw(commandBuffer, i);
    }
}

void Map::update() {

    frame = *(size_t *) sharedData->data;
    printf("Frame %zu\n", frame);

    VkRender::UBOMatrix mat{};
    mat.model = glm::mat4(1.0f);
    mat.model = glm::translate(mat.model, glm::vec3(0.0f, 0.0f, 3.0f));
    mat.model = glm::rotate(mat.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    mat.model = glm::rotate(mat.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));

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


void Map::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

    }
}
