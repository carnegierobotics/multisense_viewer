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
        Viewport,
        MultiSenseViewer,
        SceneHierarchy,
        TestWindow,
        };

    static std::vector<EditorType> getEditorTypes() {
        return {EditorType::Viewport,
                EditorType::MultiSenseViewer,
                EditorType::SceneHierarchy,
                EditorType::TestWindow};
    };
    // Function to convert enum to string
    static std::string editorTypeToString(EditorType type) {
        switch(type) {
            case EditorType::MultiSenseViewer: return "MultiSense Viewer";
            case EditorType::SceneHierarchy: return "Scene Hierarchy";
            case EditorType::TestWindow: return "Test Window";
            case EditorType::Viewport: return "Editor Viewport";
            default: return "Unknown";
        }
    }

    // Function to convert string to enum
    static EditorType stringToEditorType(const std::string &str) {
        if (str == "MultiSense Viewer") return EditorType::MultiSenseViewer;
        if (str == "Scene Hierarchy") return EditorType::SceneHierarchy;
        if (str == "Test Window") return EditorType::TestWindow;
        if (str == "Viewport") return EditorType::Viewport;
        throw std::invalid_argument("Unknown editor type string");
    }
}
#endif //MULTISENSE_VIEWER_EDITORDEFINITIONS_H
