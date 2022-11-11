//
// Created by magnus on 11/10/22.
//

#include "GroundTruthModel.h"
#include "MultiSense/Src/VO/LazyCSV.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

void GroundTruthModel::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/box.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/box.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};

    m_TruthModel = std::make_unique<glTFModel::Model>(renderUtils.device);
    m_TruthModel->loadFromFile(Utils::getAssetsPath() + "Models/Duck.gltf", renderUtils.device,
                               renderUtils.device->m_TransferQueue, 1.0f);
    m_TruthModel->createRenderPipeline(renderUtils, shaders);

    lazycsv::parser parser{"../Slam/G0/G-0_ground_truth/gt_5DoF_gnss.csv"};

    std::vector<std::string_view> coords;

    gtPositions.reserve(50000);
    for (const auto row: parser) {
        try {
            const auto [time, x, y, z, q1, q2, q3, q4] = row.cells(0, 3, 4, 5, 9, 10, 11, 12); // indexes must be in ascending order

            gtPos pos{};

            pos.x = std::stof(std::string(x.trimed()));
            pos.y = std::stof(std::string(y.trimed()));
            pos.z = std::stof(std::string(z.trimed()));
            pos.orientation.w = std::stof(std::string(q1.trimed()));
            pos.orientation.x  = std::stof(std::string(q2.trimed()));
            pos.orientation.y  = std::stof(std::string(q3.trimed()));
            pos.orientation.z  = std::stof(std::string(q4.trimed()));
            pos.time = std::stod(std::string(time.trimed()));


            gtPositions.push_back(pos);
        } catch (...) {
        }
    }

    requestAdditionalBuffers(10);

}

void GroundTruthModel::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b)
        m_TruthModel->draw(commandBuffer, i);
}//0.2306,    0.9983,   -0.0485,   -0.0122,

void GroundTruthModel::update() {
    shared = (VkRender::Shared*) sharedData->data;
    glm::mat4 rotationMat(1.0f);
    glm::mat4 translationMat(1.0f);
    if (!shared->time.empty()){
        size_t frame = shared->frame;
        double time = std::stod(shared->time) / 1000000000;

        double val = findClosest(gtPositions, gtPositions.size(), time);
        size_t index = 0;
        for (int i = 0; i < gtPositions.size(); ++i) {
            if (val == gtPositions[i].time)
                index = i;
        }
        //std::cout << time - gtPositions[index].time << std::endl;
        glm::quat rot(gtPositions[index].orientation.w,gtPositions[index].orientation.x,gtPositions[index].orientation.y,gtPositions[index].orientation.z);
        rotationMat = glm::toMat4(rot);
        translationMat = glm::translate(translationMat, glm::vec3(gtPositions[index].x, gtPositions[index].z, -gtPositions[index].y));

        //std::cout << index << std::endl;
        //std::cout << frame << std::endl;
        //std::cout << glm::to_string(rotationMat) << std::endl;
        //std::cout << glm::to_string(translationMat) << std::endl;

    }


    VkRender::UBOMatrix mat{};
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.001f, 0.001f, 0.001f));
    mat.model = translationMat * rotationMat * scale;

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


void GroundTruthModel::onUIUpdate(const VkRender::GuiObjectHandles *uiHandle) {
    for (const auto &d: uiHandle->devices) {
        if (d.state != AR_STATE_ACTIVE)
            continue;

    }
}


// Returns element closest to target in arr[]
double GroundTruthModel::findClosest(const std::vector<gtPos>& arr, size_t n, double target)
{
    // Doing binary search
    int i = 0, j = n, mid = 0;
    while (i < j) {
        mid = (i + j) / 2;

        if (arr[mid].time == target)
            return arr[mid].time;

        /* If target is less than array element,
            then search in left */
        if (target < arr[mid].time) {

            // If target is greater than previous
            // to mid, return closest of two
            if (mid > 0 && target > arr[mid - 1].time)
                return getClosest(arr[mid - 1].time,
                                  arr[mid].time, target);

            /* Repeat for left half */
            j = mid;
        }

            // If target is greater than mid
        else {
            if (mid < n - 1 && target < arr[mid + 1].time)
                return getClosest(arr[mid].time,
                                  arr[mid + 1].time, target);
            // update i
            i = mid + 1;
        }
    }

    // Only single element left after search
    return arr[mid].time;
}

// Method to compare which one is the more close.
// We find the closest by taking the difference
// between the target and both values. It assumes
// that val2 is greater than val1 and target lies
// between these two.
double GroundTruthModel::getClosest(double val1, double val2,
               double target)
{
    if (target - val1 >= val2 - target)
        return val2;
    else
        return val1;
}