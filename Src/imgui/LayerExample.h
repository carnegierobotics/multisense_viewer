//
// Created by magnus on 9/26/22.
//

#ifndef MULTISENSE_VIEWER_LAYEREXAMPLE_H
#define MULTISENSE_VIEWER_LAYEREXAMPLE_H

#include "Layer.h"


/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
class LayerExample : public VkRender::Layer {
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
    void onDetach () override {

    }
};

#endif //MULTISENSE_VIEWER_LAYEREXAMPLE_H
