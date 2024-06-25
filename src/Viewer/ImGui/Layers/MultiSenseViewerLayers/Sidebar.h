#ifndef SIDEBAR_H
#define SIDEBAR_H
#include <Viewer/ImGui/Layers/LayerSupport/Layer.h>

enum {
    MANUAL_CONNECT = 1,
    AUTO_CONNECT = 2
};

static void addPopup(VkRender::GuiObjectHandles *uiContext) {
    float popupWidth = 550.0f;
    float popupHeight = 600.0f;
    ImGui::SetNextWindowSize(ImVec2(popupWidth, popupHeight), ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, VkRender::Colors::CRLCoolGray);


    if (ImGui::BeginPopupModal("add_device_modal", nullptr,
                               ImGuiWindowFlags_NoDecoration)) {
        /** HEADER FIELD */
        ImVec2 popupDrawPos = ImGui::GetCursorScreenPos();
        ImVec2 headerPosMax = popupDrawPos;
        headerPosMax.x += popupWidth;
        headerPosMax.y += 50.0f;
        ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                  ImColor(VkRender::Colors::CRLRed), 9.0f, 0);
        popupDrawPos.y += 40.0f;
        ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                  ImColor(VkRender::Colors::CRLRed), 0.0f, 0);

        ImGui::PushFont(uiContext->info->font24);
        std::string title = "Connect to MultiSense";
        ImVec2 size = ImGui::CalcTextSize(title.c_str());
        float anchorPoint =
                (popupWidth - size.x) / 2; // Make a m_Title in center of popup window


        ImGui::Dummy(ImVec2(0.0f, size.y));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 10.0f);

        ImGui::SetCursorPosX(anchorPoint);
        ImGui::Text("%s", title.c_str());
        ImGui::PopFont();

        ImGui::Dummy(ImVec2(0.0f, 25.0f));

        /** PROFILE NAME FIELD */
        std::string profileName = "ProfileName";
        ImGui::Dummy(ImVec2(20.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
        ImGui::Text("1. Profile Name:");
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_FrameBg, VkRender::Colors::CRLDarkGray425);
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
        ImGui::Dummy(ImVec2(20.0f, 0.0f));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(popupWidth - 40.0f);
        ImGui::CustomInputTextWithHint("##InputProfileName", "MultiSense Profile", &profileName,
                                       ImGuiInputTextFlags_AutoSelectAll);
        ImGui::Dummy(ImVec2(0.0f, 30.0f));
        ImGui::PopStyleColor();


        /** SELECT METHOD FOR CONNECTION FIELD */
        ImGui::Dummy(ImVec2(20.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextGray);
        ImGui::Text("2. Select method for connection:");
        ImGui::PopStyleColor();
        ImGui::Dummy(ImVec2(0.0f, 5.0f));

        ImGui::Dummy(ImVec2(20.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(10.0f, 0.0f));

        ImVec2 uv0 = ImVec2(0.0f, 0.0f); // UV coordinates for lower-left
        ImVec2 uv1 = ImVec2(1.0f, 1.0f);
        ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // No tint

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        //ImGui::BeginChild("IconChild", ImVec2(uiContext->info->popupWidth, 40.0f), false, ImGuiWindowFlags_NoDecoration);
        ImVec2 imageButtonSize(245.0f, 55.0f);
        ImGui::PushFont(uiContext->info->font15);

        static int connectMethodSelector = 0;

        if (ImGui::ImageButtonText("Automatic", &connectMethodSelector, AUTO_CONNECT, imageButtonSize,
                                   uiContext->info->imageButtonTextureDescriptor[3], ImVec2(33.0f, 31.0f), uv0, uv1,
                                   tint_col)) {
            Log::Logger::getInstance()->info(
                "User clicked AUTO_CONNECT. Tab is {}, 0 = none, 1 = AutoConnect, 2 = ManualConnect",
                connectMethodSelector);
            uiContext->usageMonitor->userClickAction("Automatic", "ImageButtonText", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::SameLine(0, 30.0f);
        if (ImGui::ImageButtonText("Manual", &connectMethodSelector, MANUAL_CONNECT, imageButtonSize,
                                   uiContext->info->imageButtonTextureDescriptor[4], ImVec2(40.0f, 40.0f), uv0, uv1,
                                   tint_col)) {
            Log::Logger::getInstance()->info(
                "User clicked MANUAL_CONNECT. Tab is {}, 0 = none, 1 = AutoConnect, 2 = ManualConnect",
                connectMethodSelector);
            uiContext->usageMonitor->userClickAction("Manual", "ImageButtonText", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::PopFont();
        ImGui::PopStyleVar(3); // RadioButton

        /** CANCEL/CONNECT FIELD BEGINS HERE*/
        ImGui::Dummy(ImVec2(0.0f, 40.0f));
        ImGui::SetCursorPos(ImVec2(0.0f, popupHeight - 50.0f));
        ImGui::Dummy(ImVec2(20.0f, 0.0f));
        ImGui::SameLine();
        ImGui::PushFont(uiContext->info->font15);
        bool btnCancel = ImGui::Button("Close", ImVec2(190.0f, 30.0f));
        ImGui::SameLine(0, 130.0f);

        bool btnConnect = ImGui::Button("Connect", ImVec2(190.0f, 30.0f));
        ImGui::PopFont();

        if (btnCancel) {
            uiContext->usageMonitor->userClickAction("Cancel", "button",
                                                     ImGui::GetCurrentWindow()->Name);
            ImGui::CloseCurrentPopup();
        }

        if (btnConnect) {
            uiContext->usageMonitor->userClickAction("Connect", "button",
                                                     ImGui::GetCurrentWindow()->Name);
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(5); // popup style vars
}

static void addDeviceButton(VkRender::GuiObjectHandles *uiContext) {
    ImGui::SetCursorPos(ImVec2(0.0f, uiContext->info->height - 50.0f));

    ImGui::PushStyleColor(ImGuiCol_Button, VkRender::Colors::CRLBlueIsh);
    if (ImGui::Button("ADD DEVICE", ImVec2(uiContext->info->sidebarWidth, 35.0f))) {
        ImGui::OpenPopup("add_device_modal");
        uiContext->usageMonitor->userClickAction("ADD_DEVICE", "button", ImGui::GetCurrentWindow()->Name);
    }
    ImGui::PopStyleColor();
}


void drawSideBar(VkRender::GuiObjectHandles *uiContext) {
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, VkRender::Colors::CRLGray424Main);
    // Begin the sidebar as a child window
    ImGui::BeginChild("Sidebar", ImVec2(uiContext->info->sidebarWidth, uiContext->info->height), false,
                      ImGuiWindowFlags_NoScrollWithMouse);

    addPopup(uiContext);
    //askUsageLoggingPermissionPopUp(uiContext);

    // Settings button
    {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        if (ImGui::Button("Settings", ImVec2(uiContext->info->sidebarWidth, 17.0f))) {
            uiContext->showDebugWindow = !uiContext->showDebugWindow;
            uiContext->usageMonitor->userClickAction("Settings", "button", ImGui::GetCurrentWindow()->Name);
        }
        ImGui::PopStyleVar();
    }


    addDeviceButton(uiContext);


    // Add version number
    ImGui::SetCursorPos(ImVec2(0.0f, uiContext->info->height - 10.0f));
    ImGui::PushFont(uiContext->info->font8);
    ImGui::Text("%s", (std::string("Ver: ") + VkRender::RendererConfig::getInstance().getAppVersion()).c_str());
    ImGui::PopFont();

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

#endif //SIDEBAR_H
