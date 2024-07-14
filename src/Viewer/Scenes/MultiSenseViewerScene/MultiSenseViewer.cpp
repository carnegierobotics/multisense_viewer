//
// Created by mgjer on 14/07/2024.
//
#include "Viewer/Renderer/Renderer.h"

#include "Viewer/Scenes/MultiSenseViewerScene/MultiSenseViewer.h"

namespace VkRender {

    MultiSenseViewer::MultiSenseViewer(Renderer &ctx) : Scene(ctx) {

        m_context.guiManager->pushLayer("MultiSenseViewerLayer");

    }

    void MultiSenseViewer::render() {

    }

    void MultiSenseViewer::update() {

    }


}

