//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/Core/Renderer/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewer/MultiSenseViewer.h"

namespace VkRender {

    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) : Scene(ctx) {

        m_context.guiManager->pushLayer("MultiSenseViewerLayer");

    }

    void MultiSenseViewer::render() {

    }

    void MultiSenseViewer::update() {

    }


}

