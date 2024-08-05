//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_DEFAULTSCENE_H
#define MULTISENSE_VIEWER_DEFAULTSCENE_H


#include "Viewer/VkRender/Scene.h"

#include "Viewer/Scenes/Default/Scripts/MultiSense.h"
//#include "Viewer/Scenes/Default/Scripts/ImageViewer.h"

namespace VkRender {

class DefaultScene : public Scene {

    public:

        explicit DefaultScene(Renderer& ctx);

        void loadScripts();

        void loadSkybox();

        void addGuiLayers();

        void render(CommandBuffer& drawCmdBuffers) override;

        void update() override;


    private:


    };
}


#endif //MULTISENSE_VIEWER_DEFAULTSCENE_H
