#ifndef MULTISENSE_WELCOME_SCREEN_LAYER_H
#define MULTISENSE_WELCOME_SCREEN_LAYER_H

#include "Viewer/Modules/MultiSense/CommonHeader.h"
#include "Viewer/VkRender/ImGui/Layer.h"

namespace VkRender {
    class WelcomeScreenLayer : public Layer {
    public:

        /** Called once per frame **/
        void onUIRender() override {


            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;
            ImGui::SetNextWindowPos(m_editor.info->editorStartPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(m_editor.info->editorSize);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("WelcomeScreen", &pOpen, window_flags);

            ImGui::End();
            ImGui::PopStyleVar();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        }

        /** Called once upon this object creation**/
        void onAttach() override {

        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }

        /** Called once upon this object destruction **/
        void onDetach() override {
        }
    };
}

#endif //WELCOMESCREEN_H
