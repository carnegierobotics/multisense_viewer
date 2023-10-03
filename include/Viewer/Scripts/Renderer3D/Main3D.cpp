//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/Main3D.h"
#include "Viewer/ImGui/ScriptUIAddons.h"

void Main3D::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/object.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/object.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};

    KS21 = std::make_unique<GLTFModel::Model>(&renderUtils, renderUtils.device);
    KS21->loadFromFile(Utils::getAssetsPath().append("Models/ks21_pbr.gltf").string(), renderUtils.device,
                       renderUtils.device->m_TransferQueue, 1.0f);
    KS21->createRenderPipeline(renderUtils, shaders);


    Widgets::make()->inputText("Renderer3D", "##File: ", buf);
    Widgets::make()->button("Renderer3D", "Play", &play);
    Widgets::make()->button("Renderer3D", "Stop", &stop);
    Widgets::make()->button("Renderer3D", "Restart", &restart);

    lastPrintedTime = std::chrono::steady_clock::now();

}


void Main3D::update() {
    auto &d = bufferOneData;

    if (stop)
        entries.clear();

    if (play){
        std::filesystem::path path(buf);
        std::ifstream file(path);
        std::string line;

        // Skip header
        std::getline(file, line);


        while (std::getline(file, line)) {
            std::istringstream ss(line);
            Data entry;

            std::getline(ss, entry.timestamp, ',');
            entry.timePoint = convertToTimePoint(entry.timestamp);
            ss >> entry.x;
            ss.ignore();  // ignore comma
            ss >> entry.y;
            ss.ignore();  // ignore comma
            ss >> entry.z;

            entries.push_back(entry);
        }

        for (size_t i = 1; i < entries.size(); ++i) {
            entries[i].duration = entries[i].timePoint - entries[i-1].timePoint;
        }
        entries[0].duration = std::chrono::nanoseconds (0);  // First entry has no prior timestamp
    }




    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;
    d->camPos = glm::vec3(
            static_cast<double>(-renderData.camera->m_Position.z) * sin(
                    static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
            static_cast<double>(-renderData.camera->m_Position.z) * sin(
                    static_cast<double>(glm::radians(renderData.camera->m_Rotation.x))),
            static_cast<double>(renderData.camera->m_Position.z) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.y))) *
            cos(static_cast<double>(glm::radians(renderData.camera->m_Rotation.x)))
    );

    auto &d2 = bufferTwoData;
    d2->lightDir = glm::vec4(
            static_cast<double>(sinf(glm::radians(lightSource.rotation.x))) * cos(
                    static_cast<double>(glm::radians(lightSource.rotation.y))),
            sin(static_cast<double>(glm::radians(lightSource.rotation.y))),
            cos(static_cast<double>(glm::radians(lightSource.rotation.x))) * cos(
                    static_cast<double>(glm::radians(lightSource.rotation.y))),
            0.0f);


    auto *ptr = reinterpret_cast<VkRender::FragShaderParams *>(sharedData->data);
    d2->gamma = ptr->gamma;
    d2->exposure = ptr->exposure;
    d2->scaleIBLAmbient = ptr->scaleIBLAmbient;
    d2->debugViewInputs = ptr->debugViewInputs;
    d2->prefilteredCubeMipLevels = renderUtils.skybox.prefilteredCubeMipLevels;


    if (!entries.empty() && entryIdx < entries.size()){
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSinceLastPrint = now - lastPrintedTime;

        if (elapsedSinceLastPrint >= entries[entryIdx].duration) {
            // Print the entry
            std::cout << "Timestamp: " << entries[entryIdx].timePoint.time_since_epoch().count()
                      << ", x: " << entries[entryIdx].x
                      << ", y: " << entries[entryIdx].y
                      << ", z: " << entries[entryIdx].z << std::endl;
            float x = entries[entryIdx].x / 10.0f;
            float y = entries[entryIdx].z / 10.0f;
            float z = entries[entryIdx].y / 10.0f;

            d->model = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
            d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            d->model = glm::rotate(d->model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            d->model = glm::scale(d->model, glm::vec3(0.001f, 0.001f, 0.001f));

            lastPrintedTime = now;
            entryIdx++;
        }

    }



}

void Main3D::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b) {
        KS21->draw(commandBuffer, i);
    }
}