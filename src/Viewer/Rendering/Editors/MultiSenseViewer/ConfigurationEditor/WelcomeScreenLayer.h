#ifndef MULTISENSE_WELCOME_SCREEN_LAYER_H
#define MULTISENSE_WELCOME_SCREEN_LAYER_H

#include "Viewer/Modules/MultiSense/CommonHeader.h"
#include "Viewer/Rendering/ImGui/Layer.h"

namespace VkRender {
    class WelcomeScreenLayer : public Layer {
    public:

        /** Called once per frame **/
        void onUIRender() override {

            if (m_context->multiSense()->anyMultiSenseDeviceOnline())
                return;

            bool pOpen = true;
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoScrollWithMouse;
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(m_editor->ui()->width, m_editor->ui()->height));
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
