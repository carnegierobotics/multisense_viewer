//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_WELCOMESCREENLAYER_H
#define MULTISENSE_VIEWER_WELCOMESCREENLAYER_H


#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"
// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

class WelcomeScreenLayer : public VkRender::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        bool shouldDraw = true;
        for (const auto& dev: handles->devices){
            if (dev.state == CRL_STATE_ACTIVE)
                shouldDraw = false;
        }
        if (!shouldDraw || handles->renderer3D)
            return;
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("WelcomeScreen", &pOpen, window_flags);
        ImVec2 winSize = ImGui::GetWindowSize();

        ImVec2 btnSize(120.0f, 50.0f);

        // Btn X length is: 150 * 2 + 20 = 320
        // So I should start drawing at winSize / 2 - 160.0f
        ImGui::SetCursorPos(ImVec2(winSize.x / 2.0f - 160.0f, winSize.y / 2.0f - 25.0f));
        ImGui::PushFont(handles->info->font15);
        if(ImGui::Button("ADD DEVICE", btnSize)){
            handles->openAddDevicePopup = true;
            handles->usageMonitor->userClickAction("ADD_DEVICE", "button", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::SameLine(0.0f, 20.0f);
        if(ImGui::Button("3D RENDERER", btnSize)){
            // Open 3D Renderer with a basic scene.
            // Get rid of sidebar
            //
            handles->renderer3D = true;
        }
        ImGui::PopFont();
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();

    }


    /** Called once upon this object destruction **/
    void onDetach() override {

    }
};

DISABLE_WARNING_POP


#endif //MULTISENSE_VIEWER_WELCOMESCREENLAYER_H
