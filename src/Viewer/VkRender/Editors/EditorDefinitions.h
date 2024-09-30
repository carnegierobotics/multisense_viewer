//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORDEFINITIONS_H
#define MULTISENSE_VIEWER_EDITORDEFINITIONS_H

#include "Viewer/Application/pch.h"

namespace VkRender {

    // Define the EditorType enum
    enum class EditorType {
        None,
        SceneRenderer,
        Viewport3D,
        ImageEditor,
        Properties,
        GaussianViewer,
        MultiSenseViewer_Sidebar,
        MultiSenseViewer_Configuration,
        SceneHierarchy,
        TestWindow,
        };

    static std::vector<EditorType> getAllEditorTypes() {
        return {
                EditorType::SceneRenderer,
                EditorType::Viewport3D,
                EditorType::ImageEditor,
                EditorType::Properties,
                EditorType::GaussianViewer,
                EditorType::MultiSenseViewer_Sidebar,
                EditorType::MultiSenseViewer_Configuration,
                EditorType::SceneHierarchy,
                EditorType::TestWindow};
    };
    // Function to convert enum to string
    static std::string editorTypeToString(EditorType type) {
        switch(type) {
            case EditorType::SceneRenderer: return "Scene Renderer";
            case EditorType::MultiSenseViewer_Sidebar: return "MultiSense Viewer Sidebar";
            case EditorType::MultiSenseViewer_Configuration: return "MultiSense Viewer Configuration";
            case EditorType::SceneHierarchy: return "Scene Hierarchy";
            case EditorType::TestWindow: return "Test Window";
            case EditorType::Viewport3D: return "3D Viewport";
            case EditorType::Properties: return "Properties";
            case EditorType::GaussianViewer: return "Gaussian Viewer";
            case EditorType::ImageEditor: return "Image Editor";
            default: return "Unknown";
        }
    }

    // Function to convert string to enum
    static EditorType stringToEditorType(const std::string &str) {
        if (str == "Scene Renderer") return EditorType::SceneRenderer;
        if (str == "Scene Hierarchy") return EditorType::SceneHierarchy;
        if (str == "Test Window") return EditorType::TestWindow;
        if (str == "3D Viewport") return EditorType::Viewport3D;
        if (str == "Properties") return EditorType::Properties;
        if (str == "Gaussian Viewer") return EditorType::GaussianViewer;
        if (str == "Image Editor") return EditorType::ImageEditor;
        if (str == "MultiSense Viewer Sidebar") return EditorType::MultiSenseViewer_Sidebar;
        if (str == "MultiSense Viewer Configuration") return EditorType::MultiSenseViewer_Configuration;
        throw std::invalid_argument("Unknown editor type string");
    }
}
#endif //MULTISENSE_VIEWER_EDITORDEFINITIONS_H
