//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORPROPERTIES
#define MULTISENSE_VIEWER_EDITORPROPERTIES

#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {

    class EditorProperties : public Editor {
    public:
        EditorProperties() = delete;

        explicit EditorProperties(EditorCreateInfo &createInfo, UUID uuid = UUID()) : Editor(
                createInfo, uuid) {

            addUI("EditorUILayer");
            addUI("PropertiesLayer");
            addUI("DebugWindow");

        }
        void onRender(CommandBuffer &drawCmdBuffers) override {

        }

        void onUpdate() override {

        }

    };
}
#endif //MULTISENSE_VIEWER_EDITORPROPERTIES
