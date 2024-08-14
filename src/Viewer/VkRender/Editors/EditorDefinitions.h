//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORDEFINITIONS_H
#define MULTISENSE_VIEWER_EDITORDEFINITIONS_H

#include "Viewer/VkRender/pch.h"

namespace VkRender {
    // Define the EditorType enum
    enum class EditorType {
        None,
        Viewport3D,
        Properties,
        MultiSenseViewer,
        SceneHierarchy,
        TestWindow,
        MyProject,
        };

    static std::vector<EditorType> getEditorTypes() {
        return {EditorType::Viewport3D,
                EditorType::Properties,
                EditorType::MultiSenseViewer,
                EditorType::SceneHierarchy,
                EditorType::TestWindow,
                EditorType::MyProject};
    };
    // Function to convert enum to string
    static std::string editorTypeToString(EditorType type) {
        switch(type) {
            case EditorType::MultiSenseViewer: return "MultiSense Viewer";
            case EditorType::SceneHierarchy: return "Scene Hierarchy";
            case EditorType::TestWindow: return "Test Window";
            case EditorType::Viewport3D: return "3D Viewport";
            case EditorType::MyProject: return "MyProject";
            case EditorType::Properties: return "Properties";
            default: return "Unknown";
        }
    }

    // Function to convert string to enum
    static EditorType stringToEditorType(const std::string &str) {
        if (str == "MultiSense Viewer") return EditorType::MultiSenseViewer;
        if (str == "Scene Hierarchy") return EditorType::SceneHierarchy;
        if (str == "Test Window") return EditorType::TestWindow;
        if (str == "3D Viewport") return EditorType::Viewport3D;
        if (str == "MyProject") return EditorType::MyProject;
        if (str == "Properties") return EditorType::Properties;
        throw std::invalid_argument("Unknown editor type string");
    }
}
#endif //MULTISENSE_VIEWER_EDITORDEFINITIONS_H
