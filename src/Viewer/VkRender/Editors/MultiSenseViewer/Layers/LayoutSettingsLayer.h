//
// Created by magnus on 8/2/24.
//

#ifndef MULTISENSE_VIEWER_LAYOUTSETTINGSLAYER_H
#define MULTISENSE_VIEWER_LAYOUTSETTINGSLAYER_H

#include "Viewer/VkRender/ImGui/Layer.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
class LayoutSettingsLayer : public VkRender::Layer {
public:


    /** Called once upon this object creation**/
    void onAttach() override {

    }

    /** Called after frame has finished rendered **/
    void onFinishedRender() override {

    }

    /** Called once per frame **/
    void onUIRender(VkRender::GuiObjectHandles& uiContext) override {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(uiContext.info->editorStartPos + ImVec2(5, 0.0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(uiContext.info->editorSize - ImVec2(5, 5));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleColor(ImGuiCol_Border, VkRender::Colors::CRLDarkGray425);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 5.0f);
        ImGui::Begin("WelcomeScreen", &pOpen, window_flags);
        ImVec2 winSize = ImGui::GetWindowSize();

        ImGui::SetCursorPos(ImVec2(10.0f, uiContext.info->editorSize.y / 2 - 10));
        ImGui::Button("Layout 1"); ImGui::SameLine();
        ImGui::Button("Layout 2"); ImGui::SameLine();
        ImGui::Button("Layout 3"); ImGui::SameLine();
        ImGui::Dummy(ImVec2(100.0f, 0.0f)); ImGui::SameLine();
        ImGui::Button("2D"); ImGui::SameLine();
        ImGui::Button("3D"); ImGui::SameLine();




        ImGui::End();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(2);

    }

    /** Called once upon this object destruction **/
    void onDetach () override {

    }
};

#endif //MULTISENSE_VIEWER_LAYOUTSETTINGSLAYER_H
