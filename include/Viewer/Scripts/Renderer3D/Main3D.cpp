//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/Main3D.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Scripts/Private/ScriptUtils.h"

void Main3D::setup() {

    std::vector<VkPipelineShaderStageCreateInfo> shaders = {{loadShader("Scene/spv/object.vert",
                                                                        VK_SHADER_STAGE_VERTEX_BIT)},
                                                            {loadShader("Scene/spv/object.frag",
                                                                        VK_SHADER_STAGE_FRAGMENT_BIT)}};

    humvee = std::make_unique<GLTFModel::Model>(&renderUtils, renderUtils.device);
    humvee->loadFromFile(Utils::getAssetsPath().append("Models/humvee.gltf").string(), renderUtils.device,
                         renderUtils.device->m_TransferQueue, 1.0f);
    humvee->createRenderPipeline(renderUtils, shaders);


    Widgets::make()->fileDialog("Renderer3D", "Set file", filePathDialog, &openDialog);

    Widgets::make()->button("Renderer3D", "Start", &play);
    Widgets::make()->button("Renderer3D", "Pause", &pause);
    Widgets::make()->button("Renderer3D", "Stop", &restart);
    //Widgets::make()->slider("Renderer3D", "skip n", &val, 1, 50);
    //Widgets::make()->text("Renderer3D", "Sim time:");
    //Widgets::make()->text("Renderer3D", simTimeText.c_str(), "id1");

    Widgets::make()->checkbox("Renderer3D", "Realtime", &forceRealTime);


    lastEntryTime = std::chrono::steady_clock::now();


}




void Main3D::update() {
    auto &d = bufferOneData;

    if (pause)
        paused = !paused;

    if (restart) {
        entryIdx = 0;
        entries.clear();
    }

    if (play) {
        paused = false;

        std::filesystem::path path(filePathDialog);
        std::ifstream file(path);
        std::string line, dummy; // 'dummy' is used to skip unwanted fields

        // Skip header
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::istringstream ss(line);
            Data entry;

            // Skip timestamp, left_rect, disparity, lidar
            std::getline(ss, entry.timestamp, ',');  // index

            entry.timePoint = convertToTimePoint(entry.timestamp);

            ss >> entry.x;
            ss.ignore();  // ignore comma
            ss >> entry.y;
            ss.ignore();  // ignore comma
            ss >> entry.z;
            ss.ignore();  // ignore comma
            ss >> entry.qw;
            ss.ignore();  // ignore comma
            ss >> entry.qx;
            ss.ignore();  // ignore comma
            ss >> entry.qy;
            ss.ignore();  // ignore comma
            ss >> entry.qz;

            entries.push_back(entry);

            startPlay = std::chrono::steady_clock::now();
        }

        //for (size_t i = 1; i < entries.size(); ++i) {
        //    entries[i].timeDelta = entries[i].timePoint - entries[0].timePoint;
        //    entries[i].dt = entries[i].timePoint - entries[i - 1].timePoint;
        //}
        //entries[0].timeDelta = std::chrono::nanoseconds(0);  // First entry has no prior timestamp
        //lastEntryTime = std::chrono::steady_clock::now();

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


    if (!entries.empty() && entryIdx < entries.size() && !paused) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> rendererTimeDelta = now - lastEntryTime;

        if (entryIdx > static_cast<size_t>(val))
            entries[entryIdx].dt = entries[entryIdx].timePoint - entries[entryIdx - val].timePoint;


        if (rendererTimeDelta  >= entries[entryIdx].timeDelta || !forceRealTime) {


            // Print the entry
            float x = entries[entryIdx].x / 10.0f;
            float y = entries[entryIdx].y / 10.0f;
            float z = entries[entryIdx].z / 10.0f;

            float q0 = entries[entryIdx].qw;
            float q1 = entries[entryIdx].qx;
            float q2 = entries[entryIdx].qy;
            float q3 = entries[entryIdx].qz;

            glm::quat rot(q0, q1, q2, q3);

            d->model = glm::mat4(1.0f);
            // Transform the original quaternion by the rotation.
            d->model = glm::translate(d->model, glm::vec3(x, y, z));
            d->model = d->model * glm::mat4_cast(rot);

            // These rotations is just for facing the 3D model the correct way. (Y-up to Z-up)
            d->model = d->model * glm::mat4_cast(glm::quat(cosf(glm::pi<float>() / 4.0f), 0.0f, 0.0f,
                                                           -sinf(glm::pi<float>() / 4.0f))); // 90 deg around Z
            d->model = glm::rotate(d->model, glm::radians(90.0f),
                                   glm::vec3(1.0f, 0.0f, 0.0f)); // y-up to z-up 3D model.

            //d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            //d->model = glm::rotate(d->model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            d->model = glm::scale(d->model, glm::vec3(0.1f, 0.1f, 0.1f));


            double rate = entries[entryIdx].dt.count() / rendererTimeDelta.count();
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << rate;
            std::string rateStr = ss.str();
            //Widgets::make()->updateText("id1", rateStr);

            entryIdx += val;
        }

    } else if (entries.empty()) {
        d->model = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0));
        d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        d->model = glm::scale(d->model, glm::vec3(0.1f, 0.1f, 0.1f));
    }


}

void Main3D::draw(VkCommandBuffer commandBuffer, uint32_t i, bool b) {
    if (b) {
        humvee->draw(commandBuffer, i);
    }
}