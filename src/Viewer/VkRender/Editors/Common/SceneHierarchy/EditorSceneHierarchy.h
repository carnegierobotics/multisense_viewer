//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
#define MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H

#include "Viewer/VkRender/Editors/Editor.h"

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

        void onUpdate() override {

        }

    };
}
#endif //MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
