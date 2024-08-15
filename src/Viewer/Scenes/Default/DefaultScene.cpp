//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"
#include "Viewer/VkRender/Entity.h"

#include "Viewer/Scenes/Default/DefaultScene.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {


    DefaultScene::DefaultScene(Renderer &ctx) {
        m_sceneName = "Default Scene";
        Log::Logger::getInstance()->info("DefaultScene Constructor");
        auto entity = createEntity("FirstEntity");
        auto &modelComponent = entity.addComponent<OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "viking_room.obj");
        createNewCamera("DefaultCamera", 1280, 720);
        //auto &res = entity.addComponent<DefaultGraphicsPipelineComponent2>(&m_context->data(),);

        /*
          auto grid = m_context.createEntity("3DViewerGrid");
          //grid.addComponent<CustomModelComponent>(&m_context.data());

          loadSkybox();
          loadScripts();
          addGuiLayers();
          */

        auto gaussianEntity = createEntity("GaussianEntity");
        auto &gaussianEntityModelComponent = gaussianEntity.addComponent<GaussianModelComponent>(
                Utils::getModelsPath() / "3dgs" / "coordinates.ply");
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
        //auto &modelComponent = skybox.addComponent<GLTFModelComponent>(Utils::getModelsPath() / "Box" / "Box.gltf", m_context.data().device);
        //skybox.addComponent<SkyboxGraphicsPipelineComponent>(&m_context.data(), modelComponent);

    }

    void DefaultScene::addGuiLayers() {
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