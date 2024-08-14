//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_MENULAYER_H
#define MULTISENSE_VIEWER_MENULAYER_H

#include "Viewer/VkRender/ImGui/Layer.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class MenuLayer : public VkRender::Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender(VkRender::GuiObjectHandles& handles) override {


            // Push style variables
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 5.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 10.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 3.0f));

            // Set colors
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));


            ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, handles.info->menuBarHeight));
            ImGui::BeginMainMenuBar();
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Load Default Scene")) {
                    // Handle New action
                    handles.m_context->loadScene("Default Scene");
                }
                if (ImGui::MenuItem("Load Other Scene")) {
                    // Handle New action
                    handles.m_context->loadScene("MultiSense Viewer");
                }
                if (ImGui::MenuItem("Quit")) {
                    // Handle New action
                    handles.m_context->closeApplication();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Fix Aspect Ratio", nullptr, &handles.fixAspectRatio);
                ImGui::MenuItem("Revert Window Layout", nullptr, &handles.revertWindowLayout);
                ImGui::MenuItem("Console", nullptr, &handles.showDebugWindow);
                ImGui::EndMenu();
            }


            ImGui::EndMainMenuBar();


            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor(5);


        }

        /** Called once upon this object destruction **/
        void onDetach() override {

        }
    };

}
#endif //MULTISENSE_VIEWER_MENULAYER_H
