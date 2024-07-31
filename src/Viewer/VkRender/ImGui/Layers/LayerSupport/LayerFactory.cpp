//
// Created by magnus on 10/2/23.
//

#include "Viewer/VkRender/ImGui/Layers/LayerSupport/LayerFactory.h"
#include "Viewer/VkRender/ImGui/Layers/LayerSupport/LayerExample.h"
#include "Viewer/VkRender/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/VkRender/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/VkRender/Editors/Layers/EditorUILayer.h"
#include "Viewer/VkRender/Editors/Layers/SceneHierarchyLayer.h"
#include "Viewer/VkRender/Editors/Test/EditorTestLayer.h"
#include "Viewer/Scenes/Default/UILayers/SideBarLayer.h"
#include "Viewer/Scenes/Default/UILayers/MenuLayer.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/MultiSenseViewerLayer.h"

namespace VkRender {


    std::shared_ptr<Layer> LayerFactory::createLayer(const std::string &layerName) {

        if (layerName == "LayerExample") return std::make_shared<LayerExample>();
        if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
        if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
        if (layerName == "MultiSenseViewerLayer") return std::make_shared<MultiSenseViewerLayer>();
        if (layerName == "SideBarLayer") return std::make_shared<SideBarLayer>();
        if (layerName == "MenuLayer") return std::make_shared<MenuLayer>();
        if (layerName == "EditorUILayer") return std::make_shared<EditorUILayer>();
        if (layerName == "SceneHierarchyLayer") return std::make_shared<SceneHierarchyLayer>();
        if (layerName == "EditorTestLayer") return std::make_shared<EditorTestLayer>();

        throw std::runtime_error("Tried to push layer: " + layerName + " Which doesn't exists");
    }
};