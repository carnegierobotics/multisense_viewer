//
// Created by magnus on 6/27/24.
//

#ifndef CONFIGURATIONPAGE_H
#define CONFIGURATIONPAGE_H

#include "Viewer/VkRender/ImGui/Layer.h"


namespace VkRender {

    class ConfigurationLayer : public Layer {
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
            ImVec2 winSize = ImGui::GetWindowSize();

            drawConfigurationPage();


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


        void drawConfigurationPage() {
            MultiSense::MultiSenseDevice multisense{};
            /*
            auto devices = m_editor.shared->multiSenseRendererBridge->getProfileList();
            for (const auto &device: devices) {
                if (device.connectionState == MultiSense::MULTISENSE_CONNECTED) {
                    multisense = device;
                }
            }
            */
            //if (multisense.connectionState != MultiSense::MULTISENSE_CONNECTED)
            //    return;

            // Draw the page
            ImGuiWindowFlags window_flags = 0;
            window_flags =
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
/*
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(3.0f, 5.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, m_editor.info->scrollbarSize);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 0.0f);
        */
            ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 0.0f);

            ImGui::BeginChild("ControlArea",
                              ImVec2(m_editor.editorUi->width, m_editor.editorUi->height),
                              window_flags | ImGuiWindowFlags_NoBringToFrontOnFocus);

            /// DRAW EITHER 2D or 3D Control TAB. ALSO TOP TAB BARS FOR CONTROLS OR SENSOR PARAM
            ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
            ImVec2 framePadding = ImGui::GetStyle().FramePadding;
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 5.0f));
            ImGui::PushFont(m_editor.info->font15);

            float numControlTabs = 2;
            if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                /// Calculate spaces for centering tab bar text
                float tabBarWidth = m_editor.editorUi->width / numControlTabs;
                ImGui::SetNextItemWidth(tabBarWidth);
                std::string tabLabel = "Features?";
                float labelSize = ImGui::CalcTextSize(tabLabel.c_str()).x;
                float startPos = (tabBarWidth / 2) - (labelSize / 2);
                float spaceSize = ImGui::CalcTextSize(std::string(" ").c_str()).x;
                if (startPos < 1)
                    startPos = 0;
                std::string spaces(int(startPos / spaceSize), ' ');

                if (ImGui::BeginTabItem((spaces + tabLabel).c_str())) {
                    ImGui::PushFont(m_editor.info->font13);

                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);

                    ImGui::PopStyleVar(); //Framepadding
                    ImGui::PopFont();
                    ImGui::EndTabItem();
                }
                if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                    m_editor.usageMonitor->userClickAction("Preview Control", "tab",
                                                            ImGui::GetCurrentWindow()->Name);
                /// Calculate spaces for centering tab bar text
                ImGui::SetNextItemWidth(tabBarWidth);
                tabLabel = "Camera Settings";
                labelSize = ImGui::CalcTextSize(tabLabel.c_str()).x;
                startPos = (tabBarWidth / 2) - (labelSize / 2);
                if (startPos < 1)
                    startPos = 0;
                spaces = std::string(int(startPos / spaceSize), ' ');

                if (ImGui::BeginTabItem((spaces + tabLabel).c_str())) {
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);
                    ImGui::PushFont(m_editor.info->font13);

                    createCameraSettingsBar();

                    ImGui::PopStyleVar(); //Framepadding
                    ImGui::PopFont();
                    ImGui::EndTabItem();
                }
                if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                    m_editor.usageMonitor->userClickAction("Sensor Config", "tab",
                                                            ImGui::GetCurrentWindow()->Name);

                ImGui::EndTabBar();
            }
            ImGui::PopFont();
            ImGui::PopStyleVar(2); // Framepadding
            ImGui::EndChild();
        }

        void createCameraSettingsBar() {

            float textSpacing = 90.0f;
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
            static bool autoExp = true;
            // Exposure Tab
            ImGui::PushFont(m_editor.info->font18);
            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::Dummy(ImVec2(10.0f, 0.0f));
            ImGui::SameLine();
            ImGui::Text("Stereo camera");
            ImGui::PopFont();
            ImGui::SameLine(0, 60.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLBlueIsh);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5.0f);
            ImGui::Text("Hold left ctrl to type in values");
            ImGui::PopStyleColor();

            ImGui::Dummy(ImVec2(00.0f, 15.0));
            ImGui::Dummy(ImVec2(25.0f, 0.0));
            ImGui::SameLine();
            ImGui::Text("Set sensor resolution:");
            static uint32_t selectedModeIndex = 0;
            std::string resLabel = "##Resolution";
            std::vector<std::string> availableResolutions{"960x600x256", "960x600x128", "960x600x64"};
            std::string currentRes = availableResolutions[selectedModeIndex];

            ImGui::Dummy(ImVec2(25.0f, 0.0));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
            if (ImGui::BeginCombo(resLabel.c_str(),
                                  currentRes.c_str(),
                                  ImGuiComboFlags_HeightSmall)) {
                for (size_t n = 0; n < availableResolutions.size(); n++) {
                    const bool is_selected = (selectedModeIndex == n);
                    if (ImGui::Selectable(availableResolutions[n].c_str(), is_selected)) {
                        selectedModeIndex = static_cast<uint32_t>(n);
                        m_editor.usageMonitor->userClickAction("Resolution", "combo", ImGui::GetCurrentWindow()->Name);
                    }
                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::PopStyleColor(); // ImGuiCol_Text


            ImGui::Dummy(ImVec2(0.0f, 10.0f));
            ImGui::Dummy(ImVec2(25.0f, 0.0f));
            ImGui::SameLine();
            std::string txt = "Auto Exp.:";
            ImVec2 txtSize = ImGui::CalcTextSize(txt.c_str());
            ImGui::Text("%s", txt.c_str());
            ImGui::SameLine(0, textSpacing - txtSize.x);
            if (ImGui::Checkbox("##Enable Auto Exposure", &autoExp)) {
                m_editor.usageMonitor->userClickAction("Enable Auto Exposure", "Checkbox",
                                                        ImGui::GetCurrentWindow()->Name);

            }

            ImGui::PopStyleColor(); // ImGuiCol_Text

        }
    };
}

#endif //CONFIGURATIONPAGE_H
