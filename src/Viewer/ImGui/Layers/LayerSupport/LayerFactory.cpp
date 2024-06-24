//
// Created by magnus on 10/2/23.
//

#include "Viewer/ImGui/Layers/LayerSupport/LayerFactory.h"
#include "Viewer/ImGui/Layers/LayerSupport/LayerExample.h"
#include "Viewer/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/ImGui/Layers/SideBarLayer.h"
#include "Viewer/ImGui/Layers/MenuLayer.h"

namespace VkRender {


    std::shared_ptr<Layer> LayerFactory::createLayer(const std::string &layerName) {

        if (layerName == "LayerExample") return std::make_shared<LayerExample>();
        if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
        if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
        //if (layerName == "SideBarLayer") return std::make_shared<SideBarLayer>();
        //if (layerName == "MenuLayer") return std::make_shared<MenuLayer>();

        return nullptr; // or throw an exception if an unknown layer is requested
    }
};