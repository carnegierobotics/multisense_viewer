//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_SCENEHIERARCHY_H
#define MULTISENSE_VIEWER_SCENEHIERARCHY_H

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/Application/Application.h"
#include "Viewer/VkRender/Core/Entity.h"
#include "Viewer/VkRender/ImGui/LayerUtils.h"
#include "Viewer/VkRender/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/VkRender/Components/MeshComponent.h"
#include "Viewer/VkRender/Components/GaussianModelComponent.h"

namespace VkRender {


    class SceneHierarchyLayer : public Layer {
    public:
        void onAttach() override;

        void onDetach() override;

        void onUIRender() override;

        void onFinishedRender() override;

    private:
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;

        void drawCameraPanel(Entity &entity);

        void openImportFileDialog(const std::string &fileDescription, const std::vector<std::string> &type,
                                  LayerUtils::FileTypeLoadFlow flow);

        void handleSelectedFile(const LayerUtils::LoadFileInfo &loadFileInfo);

        void checkFileImportCompletion();

        void processEntities();

        void rightClickPopup();

        //Entity m_selectionContext;

        void drawEntityNode(Entity entity);
    };


}

#endif //MULTISENSE_VIEWER_SCENEHIERARCHY_H
