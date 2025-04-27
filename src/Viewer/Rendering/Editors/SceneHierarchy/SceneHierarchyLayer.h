//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_SCENEHIERARCHY_H
#define MULTISENSE_VIEWER_SCENEHIERARCHY_H

#include "Viewer/Rendering/ImGui/Layer.h"

namespace VkRender {


    class SceneHierarchyLayer : public Layer {
    public:
        void onAttach() override;

        void onDetach() override;

        void onUIRender() override;

        void onFinishedRender() override;

    private:
        void processEntities();

        void drawEntityNode(Entity entity);
    };


}

#endif //MULTISENSE_VIEWER_SCENEHIERARCHY_H
