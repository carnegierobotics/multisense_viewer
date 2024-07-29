//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORTESTLAYER_H
#define MULTISENSE_VIEWER_EDITORTESTLAYER_H

#include "Viewer/VkRender/ImGui/Layers/LayerSupport/Layer.h"
#include "Viewer/VkRender/ImGui/IconsFontAwesome6.h"

namespace VkRender {


    class EditorTestLayer : public Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles &handles) override {

            // Set window position and size
            ImVec2 window_pos = ImVec2(0.0f, 0.0f); // Position (x, y)
            ImVec2 window_size = ImVec2(handles.info->width, handles.info->height); // Size (width, height)



            // Set window flags to remove decorations
            ImGuiWindowFlags window_flags =
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoFocusOnAppearing;

            ImVec4 color;
            if (handles.editorUi->active)
                color = handles.editorUi->backgroundColorActive;
            else if (handles.editorUi->hovered)
                color = handles.editorUi->backgroundColorHovered;
            else
                color = handles.editorUi->backgroundColor;

            ImGui::PushStyleColor(ImGuiCol_WindowBg, color);
            // Set next window position and size
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(window_size, ImGuiCond_Always);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));

            // Create the parent window
            ImGui::Begin("EditorTestWindow", nullptr, window_flags);


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


        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };
}
#endif //MULTISENSE_VIEWER_EDITORTESTLAYER_H
