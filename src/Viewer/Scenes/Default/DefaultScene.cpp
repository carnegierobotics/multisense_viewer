//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"

#include "Viewer/Scenes/Default/DefaultScene.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Components/DefaultGraphicsPipelineComponent.h"

namespace VkRender {


    DefaultScene::DefaultScene(Renderer &ctx) {
        Log::Logger::getInstance()->info("DefaultScene Constructor");
        auto entity = createEntity("FirstEntity");
        auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "s30.obj");


        createNewCamera("DefaultCamera", 1280, 720);
        //auto &res = entity.addComponent<VkRender::DefaultGraphicsPipelineComponent2>(&m_context->data(),);

        /*
          auto grid = m_context.createEntity("3DViewerGrid");
          //grid.addComponent<VkRender::CustomModelComponent>(&m_context.data());

          loadSkybox();
          loadScripts();
          addGuiLayers();
          */
    }


    void DefaultScene::loadScripts() {
        //std::vector<std::string> availableScriptNames{"MultiSense", "ImageViewer"};
        /*
        std::vector<std::string> availableScriptNames{"MultiSense"};

        for (const auto &scriptName: availableScriptNames) {
            auto e = m_context.createEntity(scriptName);
            e.addComponent<ScriptComponent>(e.getName(),& m_context);
        }


        auto view = m_context.registry().view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->setup();
        }
           */
    }

    void DefaultScene::loadSkybox() {
        // Load skybox
        //auto skybox = m_context.createEntity("Skybox");
        //auto &modelComponent = skybox.addComponent<VkRender::GLTFModelComponent>(Utils::getModelsPath() / "Box" / "Box.gltf", m_context.data().device);
        //skybox.addComponent<VkRender::SkyboxGraphicsPipelineComponent>(&m_context.data(), modelComponent);

    }

    void DefaultScene::addGuiLayers() {
    }

    void DefaultScene::render(CommandBuffer &drawCmdBuffers) {


    }

    void DefaultScene::update(uint32_t frameIndex) {
        auto cameraEntity = findEntityByName("DefaultCamera");
        if (cameraEntity){
            auto& camera = cameraEntity.getComponent<CameraComponent>().camera;
            glm::mat4 invView = glm::inverse(camera.matrices.view);
            glm::vec4 cameraPos4 = invView * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            auto cameraWorldPosition = glm::vec3(cameraPos4);
        }

        auto entity = findEntityByName("FirstEntity");
        if (entity) {
            if (entity.hasComponent<TransformComponent>()) {
                auto &transform = entity.getComponent<TransformComponent>();
                transform.scale = glm::vec3(0.6f, 0.6f, 0.6f);
            }
        }

        auto view = m_registry.view<ScriptComponent>();
        for (auto entity: view) {
            auto &script = view.get<ScriptComponent>(entity);
            script.script->update();
        }

    }

    void DefaultScene::cleanUp() {
        // Delete everything

    }


}