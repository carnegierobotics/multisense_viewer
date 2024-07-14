//
// Created by magnus on 10/2/23.
//

#include "Viewer/ImGui/Layers/LayerSupport/LayerFactory.h"
#include "Viewer/ImGui/Layers/LayerSupport/LayerExample.h"
#include "Viewer/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/Scenes/DefaultScene/UILayers/SideBarLayer.h"
#include "Viewer/Scenes/DefaultScene/UILayers/MenuLayer.h"
#include "Viewer/Scenes/MultiSenseViewerScene/UILayers/MultiSenseViewerLayer.h"

namespace VkRender {


    std::shared_ptr<Layer> LayerFactory::createLayer(const std::string &layerName) {

        if (layerName == "LayerExample") return std::make_shared<LayerExample>();
        if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
        if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
        if (layerName == "MultiSenseViewerLayer") return std::make_shared<MultiSenseViewerLayer>();
        if (layerName == "SideBarLayer") return std::make_shared<SideBarLayer>();
        if (layerName == "MenuLayer") return std::make_shared<MenuLayer>();

        throw std::runtime_error("Tried to push layer: " + layerName + " Which doesn't exists");
    }
};