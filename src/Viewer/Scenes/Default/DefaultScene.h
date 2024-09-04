//
// Created by mgjer on 14/07/2024.
//

#ifndef MULTISENSE_VIEWER_DEFAULTSCENE_H
#define MULTISENSE_VIEWER_DEFAULTSCENE_H


#include "Viewer/Scenes/Scene.h"

#include "Viewer/Scenes/Default/Scripts/MultiSense.h"
//#include "Viewer/Scenes/Default/Scripts/ImageViewer.h"

namespace VkRender {

class DefaultScene : public Scene {

    public:

        explicit DefaultScene(Application& ctx, const std::string& name);

        void loadScripts();

        void loadSkybox();

        void addGuiLayers();

        void update(uint32_t i) override;

        void cleanUp();


    private:


    };
}


#endif //MULTISENSE_VIEWER_DEFAULTSCENE_H
