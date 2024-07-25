//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_EDITORUILAYER
#define MULTISENSE_VIEWER_EDITORUILAYER

#include "Viewer/VkRender/ImGui/Layers/LayerSupport/Layer.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class EditorUILayer : public Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles& handles) override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.info->width, handles.info->height); // Size (width, height)

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

            // Create the parent window
            ImGui::Begin("Example Window", nullptr, window_flags);

            /*
            // Set child window size
            // Create a child window with no decorations but with a background
            ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::CRLDarkGray425);
            ImGuiWindowFlags child_window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
            ImGui::BeginChild("Child Window", ImVec2(0.0f, 0.0f), true, child_window_flags);

            // Options for the combo box
            const char* items[] = { "UI", "MultiSense Viewer", "Scene Hierarchy" };
            static int item_current_idx = 0; // Here we store our current item index

            handles.editor.changed = false;
            // Combo box
            if (ImGui::BeginCombo("Editor Type:", items[item_current_idx])) {
                for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                    const bool is_selected = (item_current_idx == n);
                    if (ImGui::Selectable(items[n], is_selected)) {
                        item_current_idx = n;
                        handles.editor.selectedType = items[n];
                        handles.editor.changed = true;
                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            // End the child window
            ImGui::EndChild();
            ImGui::PopStyleColor();
            // End the parent window

             */
            float borderSize = 5.0f;
            ImU32 border_color = ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)); // White color

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 p1 = window_pos;
            ImVec2 p3 = ImVec2(window_pos.x + window_size.x, window_pos.y + window_size.y);

            // Define border color and thickness

            ImVec2 p2 = ImVec2(window_pos.x + window_size.x, window_pos.y + borderSize);
            draw_list->AddRectFilled(p1, p2, border_color); // Top border
            ImVec2 p4 = ImVec2(window_pos.x + borderSize, window_pos.y + window_size.y);
            draw_list->AddRectFilled(p1, p4, border_color); // Left border

            draw_list->AddRectFilled(ImVec2(window_pos.x, window_size.y - borderSize), ImVec2(window_size.x, window_size.y), border_color); // Bottom border

            draw_list->AddRectFilled(ImVec2(window_size.x - borderSize, window_pos.y), ImVec2(window_size.x, window_size.y), border_color); // Right border




            // Draw borders
            //draw_list->AddLine(p1, p2, border_color, border_thickness); // Top border
            //draw_list->AddLine(p2, p3, border_color, border_thickness); // Right border
            //draw_list->AddLine(p3, p4, border_color, border_thickness); // Bottom border
            //draw_list->AddLine(p4, p1, border_color, border_thickness); // Left border
            ImGui::End();
            ImGui::PopStyleVar(2);

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_EDITORUILAYER
