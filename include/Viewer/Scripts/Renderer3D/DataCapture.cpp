//
// Created by magnus on 10/3/23.
//

#include "Viewer/Scripts/Renderer3D/DataCapture.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/Renderer/Components/CustomModels.h"
#include "Viewer/Renderer/Components.h"
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/OBJModelComponent.h"
#include "Viewer/Renderer/Components/DefaultGraphicsPipelineComponent.h"
#include "Viewer/Renderer/Components/CameraGraphicsPipelineComponent.h"

void DataCapture::setup() {

    auto entity = m_context->createEntity("3dgs_object");
    auto &component = entity.addComponent<VkRender::OBJModelComponent>(
            Utils::getModelsPath() / "obj" / "3dgs.obj",
            m_context->renderUtils.device);
    entity.addComponent<VkRender::DefaultGraphicsPipelineComponent>(&m_context->renderUtils, component, true);
    entity.addComponent<VkRender::SecondaryRenderPassComponent>();

    {
        auto quad = m_context->createEntity("quad");
        auto &modelComponent = quad.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "quad.obj",
                m_context->renderUtils.device);

        modelComponent.objTexture.m_Descriptor = m_context->renderUtils.secondaryRenderPasses->front().depthImageInfo;
        quad.addComponent<VkRender::DefaultGraphicsPipelineComponent>(&m_context->renderUtils, modelComponent, false,
                                                                      "default2D.vert.spv", "default2D.frag.spv");
    }

    auto uuid = entity.getUUID();
    Log::Logger::getInstance()->info("Setup from {}. Created Entity {}", GetFactoryName(), uuid.operator std::string());

}


void DataCapture::update() {
    for (auto entity: m_context->m_registry.view<VkRender::CameraGraphicsPipelineComponent>()) {
        auto &resources = m_context->m_registry.get<VkRender::CameraGraphicsPipelineComponent>(entity);
        auto &objCamera = m_context->m_registry.get<VkRender::CameraComponent>(entity);
        auto &tag = m_context->m_registry.get<VkRender::TagComponent>(entity);
        for (auto &i: resources.renderData) {

            for (const auto &img: images) {
                if (tag.Tag == "Camera: " + std::to_string(img.imageID)) {
                    glm::quat orientation(img.qw, img.qx, -img.qy, -img.qz);
                    glm::mat4 rotMatrix = glm::mat4_cast(orientation);
                    glm::vec3 pos = -glm::transpose(glm::mat3_cast(orientation)) * glm::vec3(img.tx, -img.ty, -img.tz);

                    objCamera.camera.pose.q = orientation;
                    objCamera.camera.pose.pos = pos;
                    //objCamera.camera.flipZ = true;
                    objCamera.camera.m_type = VkRender::Camera::flycam;
                    objCamera.camera.updateViewMatrix();

                    glm::mat4 transMatrix = glm::translate(glm::mat4(1.0f), pos);
                    glm::mat4 transformationMatrix = transMatrix * rotMatrix;
                    transformationMatrix = glm::scale(transformationMatrix, glm::vec3(0.25f, 0.25f, 0.25f));

                    i.mvp.projection = objCamera.camera.matrices.perspective;
                    i.mvp.view = objCamera.camera.matrices.view;
                    i.mvp.model = transformationMatrix;
                    break;
                }
            }
        }
        resources.update();
    }

    auto &defaultCamera = m_context->getCamera();
    glm::mat4 invView = glm::inverse(defaultCamera.matrices.view);
    glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    auto cameraWorldPosition = glm::vec3(cameraPos4);


    auto gsMesh = m_context->findEntityByName("3dgs_object");
    if (gsMesh) {
        auto &obj = gsMesh.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (auto &i: obj.renderData) {

            i.uboMatrix.projection = defaultCamera.matrices.perspective;
            i.uboMatrix.view = defaultCamera.matrices.view;
            i.uboMatrix.model = glm::mat4(1.0f);
            i.uboMatrix.camPos = cameraWorldPosition;
        }
        obj.update();
    }
    auto quad = m_context->findEntityByName("quad");
    if (quad) {
        auto &obj = quad.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (size_t i = 0; i < obj.renderData.size(); ++i) {

            obj.renderData[i].uboMatrix.projection = defaultCamera.matrices.perspective;
            obj.renderData[i].uboMatrix.view = defaultCamera.matrices.view;
            obj.renderData[i].uboMatrix.camPos = cameraWorldPosition;

            auto model = glm::mat4(1.0f);

            float xOffsetPx = (m_context->renderData.width - 150.0) / m_context->renderData.width;

            float translationX = xOffsetPx * 2 - 1;
            float translationY = xOffsetPx * 2 - 1;

            // Apply translation after scaling
            model = glm::translate(model, glm::vec3(translationX, translationY, 0.0f));
            // Convert 300 pixels from the right edge into NDC
            float scaleX = 300.0f / m_context->renderData.width;

            model = glm::scale(model, glm::vec3(scaleX, scaleX, 1.0f)); // Uniform scaling in x and y
            //model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)); // Uniform scaling in x and y

            obj.renderData[i].uboMatrix.model = model;

        }
        obj.update();
    }

}

// Helper function to convert quaternion to rotation matrix
glm::mat4 DataCapture::quaternionToMatrix(double qw, double qx, double qy, double qz, double tx, double ty, double tz) {
    glm::quat q(qw, qx, qy, qz);
    glm::mat4 rotation = glm::toMat4(q);
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(tx, ty, tz));
    return translation * rotation; // Combine rotation and translation into one matrix
}

void DataCapture::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    if (uiHandle->m_loadColmapCameras) {

        images = loadPoses(uiHandle->m_loadColmapPosesPath);
        for (const auto &img: images) {
            std::cout << "Image ID: " << img.imageID << ", Image Name: " << img.imageName << std::endl;
            std::cout << "Quaternion: (" << img.qw << ", " << img.qx << ", " << img.qy << ", " << img.qz << ")"
                      << std::endl;
            std::cout << "Translation: (" << img.tx << ", " << img.ty << ", " << img.tz << ")" << std::endl;

            std::string tag = "Camera: " + std::to_string(img.imageID);
            auto camera = VkRender::Camera(uiHandle->info->width, uiHandle->info->height);
            // Set the camera at the colmap image position:
            camera.flipZ = true;
            auto e = uiHandle->m_context->createEntity(tag);
            auto &cameraComponent = e.addComponent<VkRender::CameraComponent>(camera);
            auto &rr = e.addComponent<VkRender::CameraGraphicsPipelineComponent>(&m_context->renderUtils);
            uiHandle->m_context->cameras[tag] = &cameraComponent.camera;
            uiHandle->m_cameraSelection.tag = tag;
        }

        uiHandle->m_loadColmapCameras = false;
    }
}

std::vector<DataCapture::ImageData> DataCapture::loadPoses(std::filesystem::path path) {

    std::vector<ImageData> images;
    std::ifstream file(path / "images.txt");
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return images;
    }
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }
        std::istringstream iss(line);
        ImageData data;
        if (!(iss >> data.imageID >> data.qw >> data.qx >> data.qy >> data.qz >> data.tx >> data.ty >> data.tz
                  >> data.cameraID >> data.imageName)) {
            std::cerr << "Failed to parse line: " << line << std::endl;
            continue;
        }
        images.push_back(data);
        // Skip the second line related to keypoints
        if (!getline(file, line)) {
            break;
        }
    }
    file.close();
    return images;

}


void DataCapture::draw(CommandBuffer *cb, uint32_t i, bool b) {
}
