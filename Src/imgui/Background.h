//
// Created by magnus on 10/19/22.
//

#ifndef MULTISENSE_VIEWER_BACKGROUND_H
#define MULTISENSE_VIEWER_BACKGROUND_H
#include "Layer.h"



class Background : public MultiSense::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(MultiSense::GuiObjectHandles *handles) override {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;

        if (handles->devices->empty()){
            ImGui::SetNextWindowSize(ImVec2(handles->info->width, handles->info->height));
        } else {
            ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth + handles->info->controlAreaWidth, handles->info->height));
        }
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoInputs;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, MultiSense::CRLCoolGray);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 10.0f));
        ImGui::Begin("BackgroundColor", &pOpen, window_flags);

        ImGui::End();

        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

    /** Called once upon this object destruction **/
    void onDetach () override {

    }
};
#endif //MULTISENSE_VIEWER_BACKGROUND_H
