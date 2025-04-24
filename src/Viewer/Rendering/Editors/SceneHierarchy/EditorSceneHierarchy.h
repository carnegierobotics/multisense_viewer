//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
#define MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H

#include "Viewer/Rendering/Editors/Editor.h"

namespace VkRender {

    class EditorSceneHierarchy : public Editor {
    public:
        EditorSceneHierarchy() = delete;

        explicit EditorSceneHierarchy(EditorCreateInfo &createInfo, UUID uuid) : Editor(createInfo, uuid) {

            addUI("EditorUILayer");
            addUI("SceneHierarchyLayer");
            addUI("DebugWindow");

        }
        void onRender(CommandBuffer &drawCmdBuffers) override {

        }

        void onFileDrop(const std::filesystem::path &path) override {
            Log::Logger::getInstance()->info("File dropped in Scene Hierarchy: {}", path.string());

        }

        void onUpdate() override {

        }

    };
}
#endif //MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
