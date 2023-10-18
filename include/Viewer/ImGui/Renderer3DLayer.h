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
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width, 50.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLDarkGray425);
        //ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        ImGui::Begin("3DTopBar", &pOpen, window_flags);



        handles->info->is3DTopBarHovered = ImGui::IsWindowHovered(
                ImGuiHoveredFlags_RootAndChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AnyWindow);

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
                    if (ImGui::Checkbox(elem.label.c_str(), elem.checkbox) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_CHECKBOX",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;

                case WIDGET_FLOAT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    if (ImGui::SliderFloat(elem.label.c_str(), elem.value, elem.minValue, elem.maxValue) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_FLOAT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_INT_SLIDER:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(200.0f);
                    if (ImGui::SliderInt(elem.label.c_str(), elem.intValue, elem.intMin, elem.intMax) &&
                        ImGui::IsItemActivated()) {
                        handles->usageMonitor->userClickAction(elem.label, "WIDGET_INT_SLIDER",
                                                               ImGui::GetCurrentWindow()->Name);
                    }
                    ImGui::PopStyleColor();
                    break;
                case WIDGET_TEXT:
                    ImGui::Text("%s", elem.label.c_str());
                    break;
                case WIDGET_INPUT_TEXT:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    ImGui::SetNextItemWidth(200.0f);
                    ImGui::InputText(elem.label.c_str(), elem.buf, 1024, 0);
                    ImGui::PopStyleColor();
                    break;
                    case WIDGET_BUTTON:
                    ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                    *elem.button = ImGui::Button(elem.label.c_str());
                    ImGui::PopStyleColor();
                    break;

                default:
                    break;
            }

        }
        ImGui::End();

        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->width - 225.0f, 50.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(225.0f, handles->info->height - 50.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));
        ImGui::Begin("Settings bar", &pOpen, window_flags);

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

        ImGui::Text("Camera: ");
        ImGui::Text("Position: (%.3f, %.3f, %.3f)",
                    static_cast<double>(handles->camera.pos.x),
                    static_cast<double>(handles->camera.pos.y),
                    static_cast<double>(handles->camera.pos.z));

        ImGui::Text("Rotation: (%.3f, %.3f, %.3f)",
                    static_cast<double>(handles->camera.rot.x),
                    static_cast<double>(handles->camera.rot.y),
                    static_cast<double>(handles->camera.rot.z));
        ImGui::End();
        ImGui::PopStyleVar();
        //ImGui::PopStyleVar();
        ImGui::PopStyleColor();

    }

    /** Called once upon this object destruction **/
    void onDetach () override {

    }
};

DISABLE_WARNING_POP


#endif //MULTISENSE_VIEWER_RENDERER3DLAYER_H
