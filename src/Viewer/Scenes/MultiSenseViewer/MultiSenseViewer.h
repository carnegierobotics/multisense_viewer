//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEVIEWER_H
#define MULTISENSE_VIEWER_MULTISENSEVIEWER_H

#include "Viewer/Core/Renderer/Scene.h"

namespace VkRender {
    class MultiSenseViewer : public Scene {

    public:
        explicit MultiSenseViewer(Renderer& ctx);

        void render() override;

        void update() override;

        ~MultiSenseViewer() override = default;
    };
}

#endif //MULTISENSE_VIEWER_MULTISENSEVIEWER_H
