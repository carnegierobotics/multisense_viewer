//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_RENDERER3DRIGHT_H
#define MULTISENSE_VIEWER_RENDERER3DRIGHT_H
#include "Viewer/ImGui/Layer.h"
/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
class Renderer3DRight : public VkRender::Layer {
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
        ImGui::SetNextWindowPos(ImVec2(handles->info->width - 300.0f, 0.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(300.0f, handles->info->height - (300 *  1 / handles->info->aspect)));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);

        ImGui::Begin("Renderer3DRight", &pOpen, window_flags);

        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);

        ImGui::Text("Camera: ");
        ImGui::Text("Position: (%.3f, %.3f, %.3f)",
                    static_cast<double>(handles->camera.pos.x),
                    static_cast<double>(handles->camera.pos.y),
                    static_cast<double>(handles->camera.pos.z));

        ImGui::Text("Up: (%.3f, %.3f, %.3f)",
                    static_cast<double>(handles->camera.up.x),
                    static_cast<double>(handles->camera.up.y),
                    static_cast<double>(handles->camera.up.z));

        ImGui::Text("Target: (%.3f, %.3f, %.3f)",
                    static_cast<double>(handles->camera.target.x),
                    static_cast<double>(handles->camera.target.y),
                    static_cast<double>(handles->camera.target.z));
        ImGui::PopStyleColor();



        ImGui::End();
        ImGui::PopStyleColor();
    }

    /** Called once upon this object destruction **/
    void onDetach () override {

    }
};
#endif //MULTISENSE_VIEWER_RENDERER3DRIGHT_H
