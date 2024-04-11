//
// Created by magnus on 10/2/23.
//

#include "Viewer/Scripts/Renderer3D/MultiSense.h"

void MultiSense::setup() {
    std::vector<VkPipelineShaderStageCreateInfo> shaders = {
            {
                    loadShader("spv/object.vert",
                               VK_SHADER_STAGE_VERTEX_BIT)
            },
            {
                    loadShader("spv/object.frag",
                               VK_SHADER_STAGE_FRAGMENT_BIT)
            }
    };


    // Load the gltf vertices/indices into vulkan
    // Load a material info
    // Load scene info
    //skybox = std::make_shared<VkRender::GLTF::Skybox>(Utils::getAssetsPath() / "Models" / "box.gltf", renderUtils.device);

    // Load vulkan render resources for gltf model
    // pipelines
    // bind renderpasses
    // Also make sure to push resources to cleanup queue if we resize or exi
    //RenderResource::GLTFModel<VkRender::GLTF::Skybox> rrSkybox(&renderUtils, skybox);

    //rrSkybox.getComponent<VkRender::GLTF::Skybox>()->draw();


    startPlay = std::chrono::steady_clock::now();
}


void MultiSense::update() {
    std::chrono::duration<float> dt = std::chrono::steady_clock::now() - startPlay;

    auto &d = ubo[0].mvp;
    d->model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 0.0f));
    d->model = glm::rotate(d->model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    // d->model = glm::scale(d->model, glm::vec3(0.1f, 0.1f, 0.1f));
    d->projection = renderData.camera->matrices.perspective;
    d->view = renderData.camera->matrices.view;

    auto &dSecondary = ubo[1].mvp;
    if (renderData.renderPassIndex == 1) {
        glm::mat4 invViewMatrix = glm::inverse(renderData.camera->matrices.view);
        glm::vec3 pos = invViewMatrix[3];
        pos.x *= -1;
        pos.y *= -1;
        invViewMatrix[3] = glm::vec4(pos, 1.0f); // Use 1.0 for the homogeneous coordinate
        dSecondary->view = glm::inverse(invViewMatrix);
        dSecondary->projection = renderData.camera->matrices.perspective;
        dSecondary->view = renderData.camera->matrices.view;

    }

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

    auto &d2 = ubo[0].fragShader;
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
}

void MultiSense::draw(CommandBuffer *commandBuffer, uint32_t i, bool b) {


}
