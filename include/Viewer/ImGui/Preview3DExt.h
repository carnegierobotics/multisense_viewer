//
// Created by magnus on 9/29/23.
//

#ifndef MULTISENSE_VIEWER_PREVIEW3DEXT_H
#define MULTISENSE_VIEWER_PREVIEW3DEXT_H

#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"

// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

class Preview3DExt  : public VkRender::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *_handles) override {
        // Create a Button

        /*
        bool clicked = ImGui::Button("Dont Click", ImVec2(150.0f, 50.0f));

        if (clicked)
            throw std::runtime_error("Dont click it");
        //demo to learn more: https://github.com/ocornut/imgui
        */

    }

    /** Called once upon this object destruction **/
    void onDetach() override {

    }
};

#endif //MULTISENSE_VIEWER_PREVIEW3DEXT_H
