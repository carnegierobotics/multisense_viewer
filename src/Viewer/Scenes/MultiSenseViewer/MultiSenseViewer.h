//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_MULTISENSEVIEWER_H
#define MULTISENSE_VIEWER_MULTISENSEVIEWER_H

#include "Viewer/VkRender/Scene.h"

namespace VkRender {
    class MultiSenseViewer : public Scene {

    public:
        explicit MultiSenseViewer(Renderer& ctx);

        void update(uint32_t i) override;

        ~MultiSenseViewer() override{
        }
    };
}

#endif //MULTISENSE_VIEWER_MULTISENSEVIEWER_H
