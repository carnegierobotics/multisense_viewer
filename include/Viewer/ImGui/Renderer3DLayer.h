//
// Created by magnus on 10/2/23.
//

#ifndef MULTISENSE_VIEWER_RENDERER3DLAYER_H
#define MULTISENSE_VIEWER_RENDERER3DLAYER_H
#include "Viewer/ImGui/Layer.h"
#include "Viewer/Tools/Macros.h"
// Dont pass on disable warnings from the example
DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/
class Renderer3DLayer : public VkRender::Layer {
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
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width, 50.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
        //ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        ImGui::Begin("3DTopBar", &pOpen, window_flags);



        if(ImGui::Button("Back", ImVec2(75.0f, 20.0f))){
            handles->renderer3D = false;
        }
        ImGui::SameLine();

        if (ImGui::Button("Settings", ImVec2(125.0f, 20.0f))) {
            handles->showDebugWindow = !handles->showDebugWindow;
            handles->usageMonitor->userClickAction("Settings", "button", ImGui::GetCurrentWindow()->Name);
        }




        for (const auto &elem: Widgets::make()->elements3D) {
            // for each element type
            ImGui::SameLine();
            switch (elem.type) {
                case WIDGET_CHECKBOX:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::Checkbox(elem.label, elem.checkbox) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_CHECKBOX",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;

                case WIDGET_FLOAT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat(elem.label, elem.value, elem.minValue, elem.maxValue) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_FLOAT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_INT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderInt(elem.label, elem.intValue, elem.intMin, elem.intMax) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_INT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_TEXT:
                    ImGui::Text("%s", elem.label);
                    break;
                case WIDGET_INPUT_TEXT:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(200.0f);
                    ImGui::InputText(elem.label, elem.buf, 1024, 0);
                    ImGui::PopStyleColor();
                    break;
                    case WIDGET_BUTTON:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    *elem.button = ImGui::Button(elem.label);
                    ImGui::PopStyleColor();
                    break;
            }

        }


        ImGui::End();
        //ImGui::PopStyleVar();
        ImGui::PopStyleColor();

    }

    /** Called once upon this object destruction **/
    void onDetach () override {

    }
};

DISABLE_WARNING_POP


#endif //MULTISENSE_VIEWER_RENDERER3DLAYER_H
