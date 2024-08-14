//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/VkRender/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"
#include "Viewer/VkRender/Components/OBJModelComponent.h"
#include "Viewer/VkRender/Entity.h"

namespace VkRender {

    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) {
        m_sceneName = "MultiSense Viewer";

        auto entity = createEntity("FirstEntity");
        auto &modelComponent = entity.addComponent<VkRender::OBJModelComponent>(
                Utils::getModelsPath() / "obj" / "s30.obj");
        createNewCamera("DefaultCamera", 1280, 720);

    }

    void MultiSenseViewer::render(CommandBuffer& drawCmdBuffers) {

    }

    void MultiSenseViewer::update(uint32_t i) {

    }


}

