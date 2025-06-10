//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITORPROPERTIES
#define MULTISENSE_VIEWER_EDITORPROPERTIES

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"

namespace VkRender {

    class EditorProperties : public Editor {
    public:
        EditorProperties() = delete;

        explicit EditorProperties(EditorCreateInfo &createInfo, UUID uuid = UUID()) : Editor(
                createInfo, uuid) {

            addUIData<EditorPathTracerLayerUI>(); // TODO remove

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
