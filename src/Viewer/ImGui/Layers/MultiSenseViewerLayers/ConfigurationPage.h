//
// Created by magnus on 6/27/24.
//

#ifndef CONFIGURATIONPAGE_H
#define CONFIGURATIONPAGE_H


namespace VkRender {
    void drawConfigurationPage(GuiObjectHandles *uiContext) {
        MultiSenseDevice multisense{};
        auto devices = uiContext->crl->getProfileList();
        for (const auto &device: devices) {
            if (device.connectionState == MULTISENSE_CONNECTED) {
                multisense = device;
            }
        }
        if (multisense.connectionState != MULTISENSE_CONNECTED)
            return;

        // Draw the page
        ImGuiWindowFlags window_flags = 0;
        window_flags =
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
/*
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(3.0f, 5.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, uiContext->info->scrollbarSize);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, VkRender::Colors::CRLCoolGray);
        ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 0.0f);
        */

        ImGui::SetCursorPos(ImVec2(uiContext->info->sidebarWidth, 0.0f));
        ImGui::BeginChild("ControlArea", ImVec2(uiContext->info->controlAreaWidth, uiContext->info->controlAreaHeight),  window_flags | ImGuiWindowFlags_NoBringToFrontOnFocus);

        /// DRAW EITHER 2D or 3D Control TAB. ALSO TOP TAB BARS FOR CONTROLS OR SENSOR PARAM
        ImGuiTabBarFlags tab_bar_flags = 0; // = ImGuiTabBarFlags_FittingPolicyResizeDown;
        ImVec2 framePadding = ImGui::GetStyle().FramePadding;
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 5.0f));
        ImGui::PushFont(uiContext->info->font15);

        float numControlTabs = 2;
        if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
            /// Calculate spaces for centering tab bar text
            float tabBarWidth = uiContext->info->controlAreaWidth / numControlTabs;
            ImGui::SetNextItemWidth(tabBarWidth);
            std::string tabLabel = "Preview Control";
            float labelSize = ImGui::CalcTextSize(tabLabel.c_str()).x;
            float startPos = (tabBarWidth / 2) - (labelSize / 2);
            float spaceSize = ImGui::CalcTextSize(std::string(" ").c_str()).x;
            std::string spaces(int(startPos / spaceSize), ' ');

            if (ImGui::BeginTabItem((spaces + tabLabel).c_str())) {
                ImGui::PushFont(uiContext->info->font13);

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);


                ImGui::PopStyleVar(); //Framepadding
                ImGui::PopFont();
                ImGui::EndTabItem();
            }
            if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                uiContext->usageMonitor->userClickAction("Preview Control", "tab",
                                                       ImGui::GetCurrentWindow()->Name);
            /// Calculate spaces for centering tab bar text
            ImGui::SetNextItemWidth(tabBarWidth);
            tabLabel = "Sensor Config";
            labelSize = ImGui::CalcTextSize(tabLabel.c_str()).x;
            startPos = (tabBarWidth / 2) - (labelSize / 2);
            spaces = std::string(int(startPos / spaceSize), ' ');

            if (ImGui::BeginTabItem((spaces + tabLabel).c_str())) {
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, framePadding);
                ImGui::PushFont(uiContext->info->font13);


                ImGui::PopStyleVar(); //Framepadding
                ImGui::PopFont();
                ImGui::EndTabItem();
            }
            if (ImGui::IsItemActivated() || ImGui::IsItemClicked())
                uiContext->usageMonitor->userClickAction("Sensor Config", "tab",
                                                       ImGui::GetCurrentWindow()->Name);

            ImGui::EndTabBar();
        }
        ImGui::PopFont();
        ImGui::PopStyleVar(); // Framepadding
        ImGui::EndChild();
    }
}
#endif //CONFIGURATIONPAGE_H
