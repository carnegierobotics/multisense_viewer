//
// Created by magnus on 10/2/23.
//

#include "Viewer/ImGui/LayerFactory.h"

#include "Viewer/ImGui/SideBarLayer.h"
#include "Viewer/ImGui/WelcomeScreenLayer.h"
#include "Viewer/ImGui/MainLayer.h"
#include "Viewer/ImGui/LayerExample.h"
#include "Viewer/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/ImGui/AdditionalWindows/CustomMetadata.h"
#include "Viewer/ImGui/Renderer3DLayer.h"

std::shared_ptr<VkRender::Layer> LayerFactory::createLayer(const std::string& layerName) {
    if (layerName == "SideBarLayer") return std::make_shared<SideBarLayer>();
    if (layerName == "WelcomeScreenLayer") return std::make_shared<WelcomeScreenLayer>();
    if (layerName == "MainLayer") return std::make_shared<MainLayer>();
    if (layerName == "LayerExample") return std::make_shared<LayerExample>();
    if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
    if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
    if (layerName == "CustomMetadata") return std::make_shared<CustomMetadata>();
    if (layerName == "Renderer3DLayer") return std::make_shared<Renderer3DLayer>();

    return nullptr; // or throw an exception if an unknown layer is requested
}