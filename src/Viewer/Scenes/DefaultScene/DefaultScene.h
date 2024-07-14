//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_DEFAULTSCENE_H
#define MULTISENSE_VIEWER_DEFAULTSCENE_H


#include "Viewer/Renderer/Scene.h"

#include "Viewer/Scenes/DefaultScene/Scripts/MultiSense.h"

namespace VkRender {

class DefaultScene : public Scene {

    public:

        explicit DefaultScene(Renderer& ctx);

        void loadScripts();

        void loadSkybox();

        void addGuiLayers();

        void render() override;

        void update() override;


    private:


    };
}


#endif //MULTISENSE_VIEWER_DEFAULTSCENE_H
