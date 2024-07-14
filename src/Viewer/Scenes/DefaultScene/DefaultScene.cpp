//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/Renderer/Renderer.h"
#include "Viewer/Renderer/Entity.h"
#include "Viewer/Renderer/Components/SkyboxGraphicsPipelineComponent.h"

#include "Viewer/Scenes/DefaultScene/DefaultScene.h"
#include "Viewer/Renderer/Components/CustomModels.h"

namespace VkRender {


    DefaultScene::DefaultScene(Renderer &ctx) : Scene(ctx) {



        // Create grid

        auto grid = m_context.createEntity("3DViewerGrid");
        grid.addComponent<VkRender::CustomModelComponent>(&m_context.renderUtils);

        loadSkybox();
        loadScripts();
        addGuiLayers();
    }


    void DefaultScene::loadScripts() {
        std::vector<std::string> availableScriptNames{{"MultiSense"}};

        for (const auto &scriptName: availableScriptNames) {
            auto e = m_context.createEntity(scriptName);
            e.addComponent<ScriptComponent>(e.getName(),& m_context);
        }

        auto view = m_context.m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->setup();
        }
    }

    void DefaultScene::loadSkybox() {
        // Load skybox
        auto skybox = m_context.createEntity("Skybox");
        auto &modelComponent = skybox.addComponent<VkRender::GLTFModelComponent>(
                Utils::getModelsPath() / "Box" / "Box.gltf", m_context.renderUtils.device);
        skybox.addComponent<VkRender::SkyboxGraphicsPipelineComponent>(&m_context.renderUtils, modelComponent);

    }

    void DefaultScene::addGuiLayers() {
        m_context.guiManager->pushLayer("SideBarLayer");
        m_context.guiManager->pushLayer("MenuLayer");
    }

    void DefaultScene::render() {

    }

    void DefaultScene::update() {

        auto &camera = m_context.getCamera();
        glm::mat4 invView = glm::inverse(camera.matrices.view);
        glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        auto cameraWorldPosition = glm::vec3(cameraPos4);

        auto skybox = m_context.findEntityByName("Skybox");
        if (skybox) {
            auto &obj = skybox.getComponent<VkRender::SkyboxGraphicsPipelineComponent>();
            // Skybox
            obj.uboMatrix.projection = camera.matrices.perspective;
            obj.uboMatrix.model = glm::mat4(glm::mat3(camera.matrices.view));
            obj.uboMatrix.model = glm::rotate(obj.uboMatrix.model, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0)); // z-up rotation

            obj.update();
        }


        auto view = m_context.m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->update();
        }

    }


}