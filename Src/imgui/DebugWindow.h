//
// Created by magnus on 10/25/22.
//

#ifndef MULTISENSE_VIEWER_DEBUGWINDOW_H
#define MULTISENSE_VIEWER_DEBUGWINDOW_H

#include "Layer.h"

class DebugWindow : public VkRender::Layer {
public:


/** Called once upon this object creation**/
    void onAttach() override {

    }

/** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

/** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        if (!handles->showDebugWindow)
            return;

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        ImGui::Begin("Debugger Window", &pOpen, window_flags);

        ImGui::ShowDemoWindow();

        ImGui::End();

    }

/** Called once upon this object destruction **/
    void onDetach() override {

    }
};

#endif //MULTISENSE_VIEWER_DEBUGWINDOW_H
