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

        EditorSceneHierarchy(const VkRenderEditorCreateInfo &createInfo, RenderUtils &utils, Renderer &ctx) : Editor(
                createInfo, utils, ctx) {

            m_guiManager->pushLayer("EditorUILayer");
            m_guiManager->pushLayer("SceneHierarchyLayer");

        }


    };
}
#endif //MULTISENSE_VIEWER_EDITORSCENEHIERARCHY_H
