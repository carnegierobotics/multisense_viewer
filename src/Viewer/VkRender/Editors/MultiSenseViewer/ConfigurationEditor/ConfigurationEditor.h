//
// Created by magnus on 9/20/24.
//

#ifndef MULTISENSE_VIEWER_CONFIGURATIONEDITOR_H
#define MULTISENSE_VIEWER_CONFIGURATIONEDITOR_H

#include "Viewer/VkRender/Editors/Editor.h"

namespace VkRender {
    class ConfigurationEditor : public Editor {
    public:
        ConfigurationEditor() = delete;

        explicit ConfigurationEditor(EditorCreateInfo &createInfo) : Editor(
                createInfo) {

            addUI("WelcomeScreenLayer");
            addUI("DebugWindow");
        }

        void onRender(CommandBuffer &drawCmdBuffers) override {

        }

        void onUpdate() override {
            // If we have a device in sidebar



        }
    private:

    };
}


#endif //MULTISENSE_VIEWER_CONFIGURATIONEDITOR_H
