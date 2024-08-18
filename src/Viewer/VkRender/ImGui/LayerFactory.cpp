//
// Created by magnus on 10/2/23.
//

#include "Viewer/VkRender/ImGui/LayerFactory.h"
#include "Viewer/VkRender/ImGui/LayerExample.h"
#include "Viewer/VkRender/ImGui/AdditionalWindows/DebugWindow.h"
#include "Viewer/VkRender/ImGui/AdditionalWindows/NewVersionAvailable.h"
#include "Viewer/VkRender/Editors/EditorUILayer.h"
#include "Viewer/VkRender/Editors/Common/SceneHierarchy//SceneHierarchyLayer.h"
#include "Viewer/VkRender/Editors/Common/Test/EditorTestLayer.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/Layers/SideBarLayer.h"
#include "Viewer/VkRender/Editors/MenuLayer.h"
#include "Viewer/VkRender/Editors/MainContextLayer.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/Layers/MultiSenseViewerLayer.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/Layers/ConfigurationLayer.h"
#include "Viewer/VkRender/Editors/MultiSenseViewer/Layers/LayoutSettingsLayer.h"
#include "Viewer/VkRender/Editors/Common/3DViewport/Editor3DLayer.h"
#include "Viewer/VkRender/Editors/Common/Properties/PropertiesLayer.h"
#include "Viewer/VkRender/Editors/Common/GaussianViewer/EditorGaussianViewerLayer.h"
#include "Viewer/VkRender/Editors/Common/ImageEditor/EditorImageLayer.h"

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
        if (layerName == "MainContextLayer") return std::make_shared<MainContextLayer>();
        if (layerName == "ConfigurationLayer") return std::make_shared<ConfigurationLayer>();
        if (layerName == "LayoutSettingsLayer") return std::make_shared<LayoutSettingsLayer>();
        if (layerName == "Editor3DLayer") return std::make_shared<Editor3DLayer>();
        if (layerName == "PropertiesLayer") return std::make_shared<PropertiesLayer>();
        if (layerName == "EditorGaussianViewerLayer") return std::make_shared<EditorGaussianViewerLayer>();
        if (layerName == "EditorImageLayer") return std::make_shared<EditorImageLayer>();

        throw std::runtime_error("Tried to push layer: " + layerName + " Which doesn't exists");
    }
};