//
// Created by magnus on 10/2/23.
//

#include "Viewer/ImGui/LayerFactory.h"

#include "Viewer/ImGui/LayerExample.h"
#include "Viewer/ImGui/SideBar.h"
#include "Viewer/ImGui/MainLayer.h"
#include "Viewer/ImGui/WelcomeScreen.h"
#include "Viewer/ImGui/Renderer3D.h"
#include "Viewer/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/ImGui/AdditionalWindows/CustomMetadata.h"

std::shared_ptr<VkRender::Layer> LayerFactory::createLayer(const std::string& layerName) {
    if (layerName == "SideBar") return std::make_shared<SideBar>();
    if (layerName == "WelcomeScreen") return std::make_shared<WelcomeScreen>();
    if (layerName == "MainLayer") return std::make_shared<MainLayer>();
    if (layerName == "LayerExample") return std::make_shared<LayerExample>();
    if (layerName == "DebugWindow") return std::make_shared<DebugWindow>();
    if (layerName == "NewVersionAvailable") return std::make_shared<NewVersionAvailable>();
    if (layerName == "CustomMetadata") return std::make_shared<CustomMetadata>();
    if (layerName == "Renderer3D") return std::make_shared<Renderer3D>();

    return nullptr; // or throw an exception if an unknown layer is requested
}