//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_EDITORUILAYER
#define MULTISENSE_VIEWER_EDITORUILAYER

#include "Viewer/VkRender/ImGui/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"
#include "EditorDefinitions.h"

#include "Viewer/Application/Application.h"

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

        void togglePopup() {
            static bool toggles[] = {true, false, false, false, false};
            // Options for the combo box
            auto items = m_context->getProjectConfig().editorTypes;
            static int item_current_idx = 0; // Here we store our current item index
            m_editor.editorUi->changed = false;

            m_editor.editorUi->changed = false;
            if (ImGui::BeginPopup("EditorSelectionPopup")) {
                ImGui::SeparatorText("Editor Types");
                for (int i = 0; i < items.size(); i++) {
                    if (ImGui::Selectable(editorTypeToString(items[i]).c_str())) {
                        item_current_idx = i;
                        m_editor.editorUi->selectedType = items[item_current_idx];
                        m_editor.editorUi->changed = true;
                    }
                }
                ImGui::MenuItem("Console", nullptr, &m_editor.showDebugWindow);
                ImGui::EndPopup();
            }
        }

        /** Called once per frame **/
        void onUIRender() override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(m_editor.info->applicationWidth, m_editor.info->applicationHeight); // Size (width, height)


            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoFocusOnAppearing |
                    ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;

            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

            // Create the parent window
            ImGui::Begin("EditorBorderWindow", nullptr, window_flags);


            // End the parent window
            float borderSize = 3.0f;

            ImVec4 color;
            if (m_editor.editorUi->active)
                color = m_editor.editorUi->backgroundColorActive;
            else if (m_editor.editorUi->hovered)
                color = m_editor.editorUi->backgroundColorHovered;
            else
                color = m_editor.editorUi->backgroundColor;

            color.w = 1.0f;
            color = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
            ImU32 border_color = ImGui::ColorConvertFloat4ToU32(color); // White color

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
            // Set window flags to remove decorations
            window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus;
            ImGui::SetNextWindowPos(ImVec2(5.0f, 5.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(window_size.x - 10.0f, 50.0f), ImGuiCond_Always);
            // Create the parent window
            //ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2.0f, 2.0f));
            auto bgColor = Colors::CRLDarkGray425;
            bgColor.w = 0.5;
            ImGui::PushStyleColor(ImGuiCol_WindowBg, bgColor);

            ImGui::Begin("EditorSelectorWindow", nullptr, window_flags);

            ImGui::PushFont(m_editor.info->fontIcons);
            ImVec2 txtSize = ImGui::CalcTextSize(ICON_FA_WINDOW_RESTORE);
            txtSize.x += 10.0f;
            if (ImGui::Button(ICON_FA_WINDOW_RESTORE, ImVec2(txtSize))) {
                ImGui::OpenPopup("EditorSelectionPopup");
            }
            ImGui::PopFont();
            togglePopup();

            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
            ImGui::End();

        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_EDITORUILAYER
