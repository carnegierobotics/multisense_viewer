//
// Created by magnus on 9/20/24.
//

#ifndef MULTISENSE_VIEWER_ADDDEVICEPOPUP_H
#define MULTISENSE_VIEWER_ADDDEVICEPOPUP_H

#include "Viewer/VkRender/ImGui/Layer.h"

namespace VkRender {

    enum {
        MANUAL_CONNECT = 1,
        AUTO_CONNECT = 2
    };


    static void addPopup(GuiObjectHandles &uiContext, Application *context) {
        float popupWidth = 550.0f;
        float popupHeight = 600.0f;
        ImGui::SetNextWindowSize(ImVec2(popupWidth, popupHeight), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, Colors::CRLCoolGray);


        if (ImGui::BeginPopupModal("add_device_modal", nullptr,
                                   ImGuiWindowFlags_NoDecoration)) {
            MultiSense::MultiSenseProfileInfo profileInfo;
            /** HEADER FIELD */
            ImVec2 popupDrawPos = ImGui::GetCursorScreenPos();
            ImVec2 headerPosMax = popupDrawPos;
            headerPosMax.x += popupWidth;
            headerPosMax.y += 50.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                      ImColor(Colors::CRLRed), 9.0f, 0);
            popupDrawPos.y += 40.0f;
            ImGui::GetWindowDrawList()->AddRectFilled(popupDrawPos, headerPosMax,
                                                      ImColor(Colors::CRLRed), 0.0f, 0);

            ImGui::PushFont(uiContext.info->font24);
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
            ImGui::PushStyleColor(ImGuiCol_Text, Colors::CRLTextGray);
            ImGui::Text("1. Profile Name:");
            ImGui::PopStyleColor();
            ImGui::PushStyleColor(ImGuiCol_FrameBg, Colors::CRLDarkGray425);
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::SetNextItemWidth(popupWidth - 40.0f);
            ImGui::CustomInputTextWithHint("##InputProfileName", "MultiSense Profile", &profileInfo.profileName,
                                           ImGuiInputTextFlags_AutoSelectAll);
            ImGui::Dummy(ImVec2(0.0f, 30.0f));
            ImGui::PopStyleColor();


            /** SELECT METHOD FOR CONNECTION FIELD */
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, Colors::CRLTextGray);
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
            //ImGui::BeginChild("IconChild", ImVec2(uiContext.info->popupWidth, 40.0f), false, ImGuiWindowFlags_NoDecoration);
            ImVec2 imageButtonSize(245.0f, 55.0f);
            ImGui::PushFont(uiContext.info->font15);

            static int connectMethodSelector = 0;

            if (ImGui::ImageButtonText("Automatic", &connectMethodSelector, AUTO_CONNECT, imageButtonSize,
                                       uiContext.info->imageButtonTextureDescriptor[3], ImVec2(33.0f, 31.0f), uv0,
                                       uv1,
                                       tint_col)) {
                Log::Logger::getInstance()->info(
                        "User clicked AUTO_CONNECT. Tab is {}, 0 = none, 1 = AutoConnect, 2 = ManualConnect",
                        connectMethodSelector);
                uiContext.usageMonitor->userClickAction("Automatic", "ImageButtonText",
                                                        ImGui::GetCurrentWindow()->Name);
            }
            ImGui::SameLine(0, 30.0f);
            if (ImGui::ImageButtonText("Manual", &connectMethodSelector, MANUAL_CONNECT, imageButtonSize,
                                       uiContext.info->imageButtonTextureDescriptor[4], ImVec2(40.0f, 40.0f), uv0,
                                       uv1,
                                       tint_col)) {
                Log::Logger::getInstance()->info(
                        "User clicked MANUAL_CONNECT. Tab is {}, 0 = none, 1 = AutoConnect, 2 = ManualConnect",
                        connectMethodSelector);
                uiContext.usageMonitor->userClickAction("Manual", "ImageButtonText",
                                                        ImGui::GetCurrentWindow()->Name);
            }
            ImGui::PopFont();
            ImGui::PopStyleVar(3); // RadioButton


            /** MANUAL_CONNECT FIELD BEGINS HERE*/
            if (connectMethodSelector == MANUAL_CONNECT) {
                // AdapterSearch Threaded operation
                // Threaded adapter search for manual connect
                {
                    ImGui::Dummy(ImVec2(0.0f, 30.0f));
                    ImGui::Dummy(ImVec2(20.0f, 0.0f));
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, Colors::CRLTextGray);
                    ImGui::Text("Camera IP:");
                    ImGui::PopStyleColor();
                    ImGui::Dummy(ImVec2(0.0f, 5.0f));
                }

                ImGui::PushStyleColor(ImGuiCol_FrameBg, Colors::CRLDarkGray425);
                ImGui::Dummy(ImVec2(20.0f, 5.0f));
                ImGui::SameLine();
                ImGui::SetNextItemWidth(popupWidth - 40.0f);
                ImGui::CustomInputTextWithHint("##inputIP", "Default: 10.66.171.21", &profileInfo.inputIP,
                                               ImGuiInputTextFlags_CharsScientific |
                                               ImGuiInputTextFlags_AutoSelectAll |
                                               ImGuiInputTextFlags_CharsNoBlank);
                ImGui::Dummy(ImVec2(0.0f, 15.0f));
                ImGui::PopStyleColor();
                /* // TODO Logic to set ethernet adapter. But is it required? Feature is only avaiable with LibMultiSense on Linux, on Windows the adapter cannot be specified
               {
                   ImGui::Dummy(ImVec2(20.0f, 0.0f));
                   ImGui::SameLine();
                   ImGui::PushStyleColor(ImGuiCol_Text, Colors::CRLTextGray);
                   ImGui::Text("Select network adapter:");
                   ImGui::PopStyleColor();
                   ImGui::Dummy(ImVec2(0.0f, 5.0f));
                   ImGui::SameLine(0.0f, 10.0f);
               }
               ImGui::Dummy(ImVec2(0.0f, 5.0f));


               static int interfaceIndex = 0;
               std::vector<std::string> interfaceNameList = uiContext.multiSenseRendererBridge->getAvailableAdapterList();
               if (interfaceNameList.empty()) {
                   interfaceNameList.emplace_back("No adapter available");
               }
               std::string previewValue = interfaceNameList[interfaceIndex];
               static ImGuiComboFlags flags = 0;
               ImGui::Dummy(ImVec2(20.0f, 5.0f));
               ImGui::SameLine();
               ImGui::SetNextItemWidth(popupWidth - 40.0f);
               ImGui::PushStyleColor(ImGuiCol_PopupBg, Colors::CRLDarkGray425);
               if (ImGui::BeginCombo("##SelectAdapter", previewValue.c_str(), flags)) {
                   for (size_t n = 0; n < interfaceNameList.size(); n++) {
                       const bool is_selected = (interfaceIndex == n);
                       if (ImGui::Selectable(interfaceNameList[n].c_str(), is_selected)) {
                           interfaceIndex = static_cast<uint32_t>(n);
                           uiContext.usageMonitor->userClickAction("SelectAdapter", "combo",
                                                                    ImGui::GetCurrentWindow()->Name);
                           uiContext.multiSenseRendererBridge->setSelectedAdapter(interfaceNameList[interfaceIndex]);
                       }
                       // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                       if (is_selected)
                           ImGui::SetItemDefaultFocus();
                   }
                   ImGui::EndCombo();
               }
               ImGui::PopStyleColor(2); // ImGuiCol_FrameBg

               profileInfo.ifName = interfaceNameList[interfaceIndex]; // interfaceIndex is static variable so will always reflect latest set value
               */
            } else if (connectMethodSelector == AUTO_CONNECT) {

                // Search with gigevision
                ImGui::Dummy(ImVec2(0.0f, 30.0f));
                ImGui::Dummy(ImVec2(20.0f, 0.0f));
                ImGui::SameLine();

                if (ImGui::Button("Enumerate")) {

                }
            }

            ////** CANCEL/CONNECT FIELD BEGINS HERE*/
            ImGui::Dummy(ImVec2(0.0f, 40.0f));
            ImGui::SetCursorPos(ImVec2(0.0f, popupHeight - 50.0f));
            ImGui::Dummy(ImVec2(20.0f, 0.0f));
            ImGui::SameLine();
            ImGui::PushFont(uiContext.info->font15);
            bool btnCancel = ImGui::Button("Close", ImVec2(190.0f, 30.0f));
            ImGui::SameLine(0, 130.0f);

            bool btnConnect = ImGui::Button("Connect", ImVec2(190.0f, 30.0f));
            ImGui::PopFont();

            if (btnCancel) {
                uiContext.usageMonitor->userClickAction("Cancel", "button",
                                                        ImGui::GetCurrentWindow()->Name);
                uiContext.shared->openAddDevicePopup = false;
                ImGui::CloseCurrentPopup();
            }

            if (btnConnect) {
                profileInfo.connectionType = MultiSense::MULTISENSE_CONNECTION_TYPE_LIBMULTISENSE;
                uiContext.usageMonitor->userClickAction("Connect", "button",
                                                        ImGui::GetCurrentWindow()->Name);

                context->multiSense()->addNewProfile(profileInfo);
                uiContext.shared->openAddDevicePopup = false;
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(5); // popup style vars
    }
}
#endif //MULTISENSE_VIEWER_ADDDEVICEPOPUP_H
