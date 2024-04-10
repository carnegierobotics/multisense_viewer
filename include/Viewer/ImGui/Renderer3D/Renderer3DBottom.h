//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_RENDERER3DBOTTOM_H
#define MULTISENSE_VIEWER_RENDERER3DBOTTOM_H

#include "Viewer/ImGui/Layer.h"
/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
class Renderer3DBottom : public VkRender::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles *handles) override {
        if (!handles->renderer3D)
            return;
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoResize;
        ImGui::SetNextWindowPos(ImVec2(300.0f, handles->info->height - 150.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - (300.0f * 2), 150.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);

        ImGui::Begin("Renderer3DBottom", &pOpen, window_flags);

        VkRender::LayerUtils::WidgetPosition pos;
        pos.paddingX = 5.0f;
        pos.maxElementWidth = 230.0f;
        VkRender::LayerUtils::createWidgets(handles, WIDGET_PLACEMENT_RENDERER3D);

        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
        handles->camera.reset = false;
        if (ImGui::RadioButton("Arcball", &handles->camera.type, 0)) {
            handles->usageMonitor->userClickAction("Arcball", "RadioButton", ImGui::GetCurrentWindow()->Name);
            handles->camera.reset = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Flycam", &handles->camera.type, 1)) {
            handles->usageMonitor->userClickAction("Flycam", "RadioButton", ImGui::GetCurrentWindow()->Name);
            handles->camera.reset = true;
        }
        ImGui::SameLine();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));
        ImGui::HelpMarker(
                "Select between arcball or flycam type. Flycam uses Arrow/WASD keys to move camera and mouse + click to rotate");
        ImGui::PopStyleVar();
        handles->camera.reset |= ImGui::Button(
                "Reset camera position"); // OR true due to resetCamera may be set by clicking radio buttons above
        if (handles->camera.reset) {
            handles->usageMonitor->userClickAction("Reset camera position", "Button",
                                                   ImGui::GetCurrentWindow()->Name);
        }
        ImGui::PopStyleColor();


        ImGui::End();
        ImGui::PopStyleColor();

    }

    /** Called once upon this object destruction **/
    void onDetach () override {

    }
};
#endif //MULTISENSE_VIEWER_RENDERER3DBOTTOM_H
