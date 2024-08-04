//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
#define MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H

#include "Viewer/VkRender/Editor.h"

namespace VkRender {

    class EditorSceneHierarchy : public Editor {
    public:
        EditorSceneHierarchy() = delete;

        explicit EditorSceneHierarchy(VulkanRenderPassCreateInfo &createInfo) : Editor(createInfo) {

            addUI("EditorUILayer");
            addUI("SceneHierarchyLayer");
            addUI("DebugWindow");

        }


    };
}
#endif //MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
