//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_EDITORUILAYER
#define MULTISENSE_VIEWER_EDITORUILAYER

#include "Viewer/VkRender/ImGui/Layers/LayerSupport/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"

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

        void togglePopup(VkRender::GuiObjectHandles &handles) {
            static bool toggles[] = {true, false, false, false, false};

            // Options for the combo box
            const char *items[] = {"UI", "MultiSense Viewer", "Scene Hierarchy"};
            static int item_current_idx = 0; // Here we store our current item index

            handles.editor.changed = false;

            if (ImGui::BeginPopup("EditorSelectionPopup")) {

                ImGui::SeparatorText("Editor Types");
                for (int i = 0; i < IM_ARRAYSIZE(items); i++)
                    if (ImGui::Selectable(items[i])) {
                        item_current_idx = i;
                        handles.editor.selectedType = items[item_current_idx];
                        handles.editor.changed = true;
                    }
                ImGui::EndPopup();

            }
        }

        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.info->width, handles.info->height); // Size (width, height)

            std::vector<ImVec4> colors{{1.0f, 0.0f, 0.0f, 0.7f},
                                       {0.0f, 1.0f, 0.0f, 0.7f}, // 1: Green
                                       {0.0f, 0.0f, 1.0f, 0.7f}, // 2: Blue
                                       {1.0f, 1.0f, 0.0f, 0.7f}, // 3: Yellow
                                       {1.0f, 0.0f, 1.0f, 0.7f}, // 4: Magenta
                                       {0.0f, 1.0f, 1.0f, 0.7f}, // 5: Cyan
                                       {0.5f, 0.5f, 0.5f, 0.7f}}; // 6: Gray

            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoFocusOnAppearing |
                    ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;


            ImGui::PushStyleColor(ImGuiCol_WindowBg, colors[handles.editorIndex % 7]);
            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

            // Create the parent window
            ImGui::Begin("EditorBorderWindow", nullptr, window_flags);


            // End the parent window
            float borderSize = 3.0f;
            ImU32 border_color = ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)); // White color

            ImDrawList *draw_list = ImGui::GetWindowDrawList();
            ImVec2 p1 = window_pos;
            ImVec2 p2 = ImVec2(window_pos.x + window_size.x, window_pos.y + borderSize);
            draw_list->AddRectFilled(p1, p2, border_color); // Top border
            ImVec2 p4 = ImVec2(window_pos.x + borderSize, window_pos.y + window_size.y);
            draw_list->AddRectFilled(p1, p4, border_color); // Left border

            draw_list->AddRectFilled(ImVec2(window_pos.x, window_size.y - borderSize),
                                     ImVec2(window_size.x, window_size.y), border_color); // Bottom border

            draw_list->AddRectFilled(ImVec2(window_size.x - borderSize, window_pos.y),
                                     ImVec2(window_size.x, window_size.y), border_color); // Right border

            ImGui::End();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
            // Set window flags to remove decorations
            window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus;
            ImGui::SetNextWindowPos(ImVec2(5.0f, 5.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(window_size.x - 10.0f, 50.0f), ImGuiCond_Always);
            // Create the parent window
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 2.0f));

            ImGui::Begin("EditorSelectorWindow", nullptr, window_flags);

            ImGui::PushFont(handles.info->fontIcons);
            ImVec2 txtSize = ImGui::CalcTextSize(ICON_FA_WINDOW_RESTORE);
            txtSize.x += 10.0f;
            if (ImGui::Button(ICON_FA_WINDOW_RESTORE, ImVec2(txtSize))) {
                ImGui::OpenPopup("EditorSelectionPopup");
            }
            ImGui::PopFont();
            togglePopup(handles);

            ImGui::PopStyleVar(3);

            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_EDITORUILAYER