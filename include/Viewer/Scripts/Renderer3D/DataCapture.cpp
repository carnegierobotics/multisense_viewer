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

    std::string tag = "Camera: Test";
    auto camera = VkRender::Camera(m_context->renderData.width, m_context->renderData.height);
    // Set the camera at the colmap image position:
    auto e = m_context->createEntity(tag);
    auto &cameraComponent = e.addComponent<VkRender::CameraComponent>(camera);
    auto &rr = e.addComponent<VkRender::CameraGraphicsPipelineComponent>(&m_context->renderUtils);
    m_context->cameras[tag] = &cameraComponent.camera; // TODO I should't have to set this for the app not to crash

}


void DataCapture::update() {
    auto &defaultCamera = m_context->getCamera();

    for (auto entity: m_context->m_registry.view<VkRender::CameraGraphicsPipelineComponent>()) {
        auto &resources = m_context->m_registry.get<VkRender::CameraGraphicsPipelineComponent>(entity);
        auto &objCamera = m_context->m_registry.get<VkRender::CameraComponent>(entity);
        auto &tag = m_context->m_registry.get<VkRender::TagComponent>(entity);

        for (auto &i: resources.renderData) {

            for (const auto &img: images) {
                if (tag.Tag == "Camera: " + std::to_string(img.imageID)) {
                    glm::quat orientation(img.qw, img.qx, img.qy, img.qz);
                    orientation = glm::normalize(orientation);
                    glm::mat4 rotMatrix = glm::mat4_cast(orientation);
                    glm::vec3 cameraCenterProjection = -glm::transpose(glm::mat3_cast(orientation)) * glm::vec3(img.tx, img.ty, img.tz);

                    auto worldToCameraColmap = glm::translate(rotMatrix, cameraCenterProjection);

                    // Adjust matrix for Blender
                    glm::mat4 blenderAdjustment = glm::mat4(
                            glm::vec4(1, 0, 0, 0),
                            glm::vec4(0, 0, -1, 0),  // Swap Y and Z, negate new Y
                            glm::vec4(0, 1, 0, 0),
                            glm::vec4(0, 0, 0, 1)
                    );
                    glm::mat4 worldToCameraBlender = blenderAdjustment * worldToCameraColmap;

                    worldToCameraBlender = glm::scale(worldToCameraBlender, glm::vec3(0.25f, 0.25f, 0.25f));
                    i.mvp.projection = defaultCamera.matrices.perspective;
                    i.mvp.view = defaultCamera.matrices.view;
                    i.mvp.model = worldToCameraBlender;
                    objCamera.camera.matrices.view = glm::inverse(worldToCameraBlender);
                    /*
                    glm::quat q = glm::quat_cast(worldToCameraColmap);
                    objCamera.camera.pose.q = q;
                    objCamera.camera.pose.pos = cameraCenterProjection;
                    objCamera.camera.updateViewMatrix();
                    */
                    break;
                }
            }
        }
        resources.update();
    }

    glm::mat4 invView = glm::inverse(defaultCamera.matrices.view);
    glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    auto cameraWorldPosition = glm::vec3(cameraPos4);


    auto gsMesh = m_context->findEntityByName("3dgs_object");
    if (gsMesh) {
        auto &obj = gsMesh.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (auto &i: obj.renderData) {

            i.uboMatrix.projection = defaultCamera.matrices.perspective;
            i.uboMatrix.view = defaultCamera.matrices.view;
            auto model = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            i.uboMatrix.model = model;
            //i.uboMatrix.model = glm::mat4(1.0f);
            i.uboMatrix.camPos = cameraWorldPosition;
        }
        obj.update();
    }
    auto test_camera = m_context->findEntityByName("Camera: Test");
    if (test_camera) {
        auto &obj = test_camera.getComponent<VkRender::CameraGraphicsPipelineComponent>();
        auto &objCamera = test_camera.getComponent<VkRender::CameraComponent>();

        for (auto &i: obj.renderData) {

            i.mvp.projection = defaultCamera.matrices.perspective;
            i.mvp.view = defaultCamera.matrices.view;
            glm::mat4 model = glm::translate(glm::mat4(1.0f), objCamera.camera.pose.pos);

            model = model * glm::mat4_cast(objCamera.camera.pose.q);
            model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
            i.mvp.model = model;
            i.mvp.camPos = cameraWorldPosition;
        }
        obj.update();
    }

    auto quad = m_context->findEntityByName("quad");
    if (quad) {
        auto &obj = quad.getComponent<VkRender::DefaultGraphicsPipelineComponent>();
        for (auto &i: obj.renderData) {

            i.uboMatrix.projection = defaultCamera.matrices.perspective;
            i.uboMatrix.view = defaultCamera.matrices.view;
            i.uboMatrix.camPos = cameraWorldPosition;

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

            i.uboMatrix.model = model;

        }
        obj.update();
    }

}


bool DataCapture::compareImageID(const ImageData &img1, const ImageData &img2) {
    return img1.imageID < img2.imageID;
}

std::vector<DataCapture::CameraData> DataCapture::loadCameraParams(const std::filesystem::path &path) {
    std::vector<CameraData> cameras;
    std::ifstream file(path / "cameras.txt");
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return cameras;
    }

    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }

        std::istringstream iss(line);
        CameraData camera;
        iss >> camera.cameraID >> camera.model >> camera.width >> camera.height;

        // Since the number and type of parameters vary by camera model,
        // we will read the rest of the line as a string and then parse it into doubles.
        std::string params;
        std::getline(iss, params);
        std::istringstream paramsStream(params);
        double param;
        while (paramsStream >> param) {
            camera.parameters.push_back(param);
        }

        cameras.push_back(camera);
    }

    file.close();
    return cameras;
}

void DataCapture::onUIUpdate(VkRender::GuiObjectHandles *uiHandle) {
    if (uiHandle->m_cameraSelection.selected) {
        if (!images.empty() && uiHandle->m_cameraSelection.currentItemSelected < images.size())
            Log::Logger::getInstance()->info("Selected camera {}, Image Filename: {}",
                                             images.rbegin()[uiHandle->m_cameraSelection.currentItemSelected].imageID,
                                             images.rbegin()[uiHandle->m_cameraSelection.currentItemSelected].imageName.c_str());
    }
    if (uiHandle->m_loadColmapCameras) {
        // Load camera params
        std::vector<DataCapture::CameraData> cameras = loadCameraParams(uiHandle->m_loadColmapPosesPath);

        for (const auto &cam: cameras) {
            std::cout << "Camera ID: " << cam.cameraID << ", Model: " << cam.model
                      << ", Width: " << cam.width << ", Height: " << cam.height << std::endl;
            std::cout << "Parameters: ";
            for (auto param: cam.parameters) {
                std::cout << param << " ";
            }
            std::cout << std::endl;
        }
        // Load poses
        images = loadPoses(uiHandle->m_loadColmapPosesPath);
        std::sort(images.begin(), images.end(), DataCapture::compareImageID);
        CameraData cameraData = cameras.front();
        for (const auto &img: images) {
            std::cout << "Image ID: " << img.imageID << ", Image Name: " << img.imageName << std::endl;
            std::cout << "Quaternion: (" << img.qw << ", " << img.qx << ", " << img.qy << ", " << img.qz << ")"
                      << std::endl;
            std::cout << "Translation: (" << img.tx << ", " << img.ty << ", " << img.tz << ")" << std::endl;

            std::string tag = "Camera: " + std::to_string(img.imageID);

            float fov_x = 2.0f * atan(cameraData.width / (2.0f * cameraData.parameters[0]));
            float fov_y = 2.0f * atan(cameraData.height / (2.0f * cameraData.parameters[1]));

            // Convert to degrees
            float fov_x_deg = glm::degrees(fov_x);
            float fov_y_deg = glm::degrees(fov_y);

            // Usually, we select the smaller FOV to avoid distortion
            float fov = std::min(fov_x_deg, fov_y_deg);

            // Aspect ratio
            float aspect = static_cast<float>(cameraData.width) / static_cast<float>(cameraData.height);

            auto camera = VkRender::Camera(cameraData.width, cameraData.height);

            // Update the perspective of the camera
            camera.setPerspective(fov, aspect, 0.1f, 100.0f);

            // Set the camera at the colmap image position:
            auto e = uiHandle->m_context->createEntity(tag);
            auto &cameraComponent = e.addComponent<VkRender::CameraComponent>(camera);
            auto &rr = e.addComponent<VkRender::CameraGraphicsPipelineComponent>(&m_context->renderUtils);
            m_context->cameras[tag] = &cameraComponent.camera;
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
