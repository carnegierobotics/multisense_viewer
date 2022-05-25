//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_SIDEBAR_H
#define MULTISENSE_SIDEBAR_H


#include <algorithm>
#include "imgui_internal.h"
#include "imgui.h"
#include "Layer.h"

class SideBar : public Layer {
public:

    // Create global object for convenience in other functions
    GuiObjectHandles *handles;

    void onFinishedRender() override {

    }

    void OnUIRender(GuiObjectHandles *_handles) override {
        this->handles = _handles;
        GuiLayerUpdateInfo *info = handles->info;


        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth, info->height));
        ImGui::Begin("SideBar", &pOpen, window_flags);


        auto *wnd = ImGui::FindWindowByName("GUI");
        if (wnd) {
            ImGuiDockNode *node = wnd->DockNode;
            if (node)
                node->WantHiddenTabBarToggle = true;

        }

        ImGui::TextUnformatted(info->title.c_str());
        ImGui::TextUnformatted(info->deviceName.c_str());

        // Update frame time display
        if (info->firstFrame) {
            std::rotate(info->frameTimes.begin(), info->frameTimes.begin() + 1,
                        info->frameTimes.end());
            float frameTime = 1000.0f / (info->frameTimer * 1000.0f);
            info->frameTimes.back() = frameTime;
            if (frameTime < info->frameTimeMin) {
                info->frameTimeMin = frameTime;
            }
            if (frameTime > info->frameTimeMax) {
                info->frameTimeMax = frameTime;
            }
        }

        ImGui::PlotLines("Frame Times", &info->frameTimes[0], 50, 0, "", info->frameTimeMin,
                         info->frameTimeMax, ImVec2(0, 80));

        if (ImGui::BeginPopupModal("add_device_modal", NULL,
                                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {

            bool deviceAlreadyExist = false;
            static char inputName[32] = "Front Camera";
            static char inputIP[32] = "10.66.171.21";

            ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyResizeDown;
            if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags)) {
                if (ImGui::BeginTabItem("Select Premade Profile")) {
                    ImGui::Text("Connect to your MultiSense Device");
                    ImGui::Separator();
                    ImGui::InputText("Profile name", inputName,
                                     IM_ARRAYSIZE(inputName));
                    ImGui::InputText("Camera ip", inputIP, IM_ARRAYSIZE(inputIP));

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Advanced Options")) {
                    ImGui::Text("Connect to your MultiSense Device");
                    ImGui::Separator();
                    ImGui::InputText("Profile name", inputName,
                                     IM_ARRAYSIZE(inputName));
                    ImGui::InputText("Camera ip", inputIP, IM_ARRAYSIZE(inputIP));

                    btnConnect = ImGui::Button("connect", ImVec2(175.0f, 30.0f));
                    ImGui::SameLine();

                    // On connect button click
                    if (btnConnect) {
                        for (auto &d: devices) {
                            if (d.IP == inputIP)
                                deviceAlreadyExist = true;
                        }

                        if (!deviceAlreadyExist)
                            createNewElement(inputName, inputIP);

                        ImGui::CloseCurrentPopup();

                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }


            ImGui::EndPopup();

        }

        ImGui::Spacing();
        ImGui::Spacing();

        if (!devices.empty())
            sidebarElements();

        addDeviceButton();
        ImGui::End();

    }

private:

    std::vector<Element> devices;

    bool btnConnect = false;
    bool btnAdd = false;

    void createNewElement(char *name, char *ip) {
        Element el;

        el.name = name;
        el.IP = ip;
        el.state = ArConnectingState;

        devices.emplace_back(el);

        handles->devices = &devices;

    }

    void sidebarElements() {
        for (int i = 0; i < devices.size(); ++i) {
            auto &e = devices[i];

            std::string buttonIdentifier = "";
            // Set colors based on state
            switch (e.state) {

                case ArConnectedState:
                    break;
                case ArConnectingState:
                    buttonIdentifier = "Connecting";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.19f, 0.33f, 0.48f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.98f, 0.65f, 0.00f, 1.0f));
                    break;
                case ArActiveState:
                    buttonIdentifier = "Active";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.19f, 0.33f, 0.48f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.26f, 0.42f, 0.31f, 1.0f));
                    break;
                case ArInActiveState:
                    buttonIdentifier = "Inactive";
                    break;
                case ArDisconnectedState:
                    break;
                case ArUnavailableState:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    buttonIdentifier = "Unavailable";
                    break;
            }

            ImGui::SetCursorPos(ImVec2(0, ImGui::GetCursorPosY()));

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            std::string winId = e.IP + "Child";
            ImGui::BeginChild(winId.c_str(), ImVec2(handles->info->sidebarWidth, handles->info->elementHeight),
                              false, ImGuiWindowFlags_NoDecoration);

            ImGui::Dummy(ImVec2(0.0f, 20.0f));

            ImVec2 window_pos = ImGui::GetWindowPos();
            ImVec2 window_size = ImGui::GetWindowSize();
            ImVec2 window_center = ImVec2(window_pos.x + window_size.x * 0.5f, window_pos.y + window_size.y * 0.5f);
            ImVec2 cursorPos = ImGui::GetCursorPos();


            // Profile Name
            ImGui::PushFont(handles->info->font24);
            ImVec2 lineSize = ImGui::CalcTextSize(e.name.c_str());
            cursorPos.x = window_center.x - (lineSize.x / 2);
            ImGui::SetCursorPos(cursorPos);


            ImGui::Text("%s", e.name.c_str());
            ImGui::PopFont();


            // Camera IP Address
            ImGui::PushFont(handles->info->font13);
            lineSize = ImGui::CalcTextSize(e.IP.c_str());
            cursorPos.x = window_center.x - (lineSize.x / 2);
            ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));

            ImGui::Text("%s", e.IP.c_str());
            ImGui::PopFont();

            // Status Button
            ImGui::Dummy(ImVec2(0.0f, 5.0f));
            ImGui::PushFont(handles->info->font18);
            //ImGuiStyle style = ImGui::GetStyle();
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);

            cursorPos.x = window_center.x - (ImGui::GetFontSize() * 10 / 2);
            ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));


            buttonIdentifier += "##" + e.IP;

            e.clicked = ImGui::Button(buttonIdentifier.c_str(),
                                      ImVec2(ImGui::GetFontSize() * 10, ImGui::GetFontSize() * 2));

            ImGui::PopFont();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();

            ImGui::EndChild();

            ImGui::PopStyleColor();
            ImGui::PopStyleVar();


        }
    }

    void addDeviceButton() {

        ImGui::SetCursorPos(ImVec2(20, 650));
        btnAdd = ImGui::Button("ADD DEVICE", ImVec2(200.0f, 35.0f));

        if (btnAdd) {
            ImGui::OpenPopup("add_device_modal");

        }
    }


};


#endif //MULTISENSE_SIDEBAR_H
