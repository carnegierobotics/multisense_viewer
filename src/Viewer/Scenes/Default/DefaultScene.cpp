//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"

#include "Viewer/Scenes/Default/DefaultScene.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"

namespace VkRender {


    DefaultScene::DefaultScene(Renderer &ctx) : Scene(ctx) {


        auto entity = m_context.createEntity("FirstEntity");
        auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(Utils::getModelsPath() / "obj" / "s30.obj", &m_context.vkDevice());
        //auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->data(),);


        auto grid = m_context.createEntity("3DViewerGrid");
        //grid.addComponent<VkRender::CustomModelComponent>(&m_context.data());

        loadSkybox();
        loadScripts();
        addGuiLayers();
    }


    void DefaultScene::loadScripts() {
        //std::vector<std::string> availableScriptNames{"MultiSense", "ImageViewer"};
        /*
        std::vector<std::string> availableScriptNames{"MultiSense"};

        for (const auto &scriptName: availableScriptNames) {
            auto e = m_context.createEntity(scriptName);
            e.addComponent<ScriptComponent>(e.getName(),& m_context);
        }
        */

        auto view = m_context.registry().view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->setup();
        }
    }

    void DefaultScene::loadSkybox() {
        // Load skybox
        auto skybox = m_context.createEntity("Skybox");
        //auto &modelComponent = skybox.addComponent<VkRender::GLTFModelComponent>(Utils::getModelsPath() / "Box" / "Box.gltf", m_context.data().device);
        //skybox.addComponent<VkRender::SkyboxGraphicsPipelineComponent>(&m_context.data(), modelComponent);

    }

    void DefaultScene::addGuiLayers() {

    }

    void DefaultScene::render(CommandBuffer &drawCmdBuffers) {
        uint32_t currentFrame = *drawCmdBuffers.frameIndex;
        /*
        auto sky = m_context.findEntityByName("Skybox");
        if (sky)
            sky.getComponent<VkRender::SkyboxGraphicsPipelineComponent>().draw(&drawCmdBuffers, currentFrame);
        auto modelComponent = m_context.findEntityByName("Skybox");
        if (modelComponent)
            modelComponent.getComponent<VkRender::GLTFModelComponent>().model->draw(drawCmdBuffers.buffers[currentFrame]);

         */

        auto entity = m_context.findEntityByName("FirstEntity");
        if (entity){
            auto &resources = entity.getComponent<DefaultGraphicsPipelineComponent>();
            resources.draw(drawCmdBuffers);

        }

    }

    void DefaultScene::update() {

        auto &camera = m_context.getCamera();
        glm::mat4 invView = glm::inverse(camera.matrices.view);
        glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        auto cameraWorldPosition = glm::vec3(cameraPos4);

        auto skybox = m_context.findEntityByName("Skybox");
        if (skybox) {
            /*
            auto &obj = skybox.getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
            // Skybox
            obj.uboMatrix.projection = camera.matrices.perspective;
            obj.uboMatrix.model = glm::mat4(glm::mat3(camera.matrices.view));
            obj.uboMatrix.model = glm::rotate(obj.uboMatrix.model, glm::radians(90.0f),
                                              glm::vec3(1.0, 0.0, 0.0)); // z-up rotation

            obj.update();
*/

            auto entity = m_context.findEntityByName("FirstEntity");
            if (entity){
                    auto &resources = entity.getComponent<DefaultGraphicsPipelineComponent>();
                    auto &transform = entity.getComponent<TransformComponent>();
                    transform.scale = glm::vec3(0.6f, 0.6f, 0.6f);
                    resources.updateTransform(transform);
                    resources.updateView(camera);
                    resources.update(m_context.currentFrameIndex());
            }
        }


        auto view = m_context.registry().view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->update();
        }

    }


}