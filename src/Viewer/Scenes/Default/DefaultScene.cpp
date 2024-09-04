//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Core/Entity.h"

#include "Viewer/Scenes/Default/DefaultScene.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/RenderPipelines/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {


    DefaultScene::DefaultScene(Application &ctx, const std::string& name) : Scene(name) {
        Log::Logger::getInstance()->info("DefaultScene Constructor");
        auto entity = createEntity("FirstEntity");
        auto &modelComponent = entity.addComponent<MeshComponent>(Utils::getModelsPath() / "obj" / "viking_room.obj");
        createNewCamera("DefaultCamera", 1280, 720);
        //auto &res = entity.addComponent<DefaultGraphicsPipelineComponent2>(&m_context->data(),);

        /*
          auto grid = m_context.createEntity("3DViewerGrid");
          //grid.addComponent<CustomModelComponent>(&m_context.data());

          loadSkybox();
          loadScripts();
          addGuiLayers();
          */

    }


    void DefaultScene::loadScripts() {

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