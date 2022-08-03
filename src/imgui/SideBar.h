//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_SIDEBAR_H
#define MULTISENSE_SIDEBAR_H

#include "AutoConnect.h"

#ifdef WIN32
#include "AutoConnectWindows.h"
#else

#include "AutoConnectLinux.h"

#endif


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

    void autoDetectCamera() {

        Log::Logger::getInstance()->info("Start looking for ethernet adapters");
#ifdef WIN32
        AutoConnectWindows connect{};
    std::vector<AutoConnect::AdapterSupportResult> res =  connect.findEthernetAdapters();
#else
        AutoConnectLinux connect{};
        std::vector<AutoConnect::AdapterSupportResult> res = connect.findEthernetAdapters();
#endif

        connect.start(res);

        // TODO DONT BLOCK UI THREAD WHILE LOOKING FOR ADAPTERS
        while (true) {
            if (connect.isConnected())
                break;
        }
        connect.stop();
        AutoConnect::Result result = connect.getResult();
        crl::multisense::Channel *ptr = connect.getCameraChannel();
        crl::multisense::system::DeviceInfo info;
        ptr->getDeviceInfo(info);
        Log::Logger::getInstance()->info(
                "AUTOCONNECT: Found Camera on IP: %s, using Adapter: %s, adapter long name: %s, Camera returned name %s",
                result.cameraIpv4Address.c_str(), result.networkAdapter.c_str(), result.networkAdapterLongName.c_str(),
                info.name.c_str());

        inputIP = result.cameraIpv4Address;
        inputName = info.name;

        if (inputName == "Multisense S30")
            presetItemIdIndex = 1;
        else if(inputName == "MultiSense S21")
            presetItemIdIndex = 2;
        else
            presetItemIdIndex = 0;

    }

    void OnUIRender(GuiObjectHandles *_handles) override {
        this->handles = _handles;
        GuiLayerUpdateInfo *info = handles->info;


        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->sidebarWidth, info->height));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.031, 0.078, 0.129, 1.0f));

        ImGui::Begin("SideBar", &pOpen, window_flags);


        auto *wnd = ImGui::FindWindowByName("GUI");
        /*
        if (wnd) {
            ImGuiDockNode *node = wnd->DockNode;
            if (node)
                node->WantHiddenTabBarToggle = true;

        }
         */

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

        ImGui::SetNextWindowSize(ImVec2(info->popupWidth, info->popupHeight), ImGuiCond_Once);

        if (ImGui::BeginPopupModal("add_device_modal", NULL,
                                   ImGuiWindowFlags_NoTitleBar)) {

            bool deviceAlreadyExist = false;
            //static char inputName[32] = "Profile #1";

            ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyResizeDown;
            if (ImGui::BeginTabBar("MyTabBar", tab_bar_flags)) {
                if (ImGui::BeginTabItem("Select Premade Profile")) {
                    ImGui::Text("Connect to your MultiSense Device");
                    ImGui::Separator();
                    ImGui::InputText("Profile name##1", inputName.data(),
                                     inputFieldNameLength);

                    const char *items[] = {"Select Preset","MultiSense S30", "MultiSense S21", "Virtual Camera"};
                    const char *combo_preview_value = items[presetItemIdIndex];  // Pass in the preview value visible before opening the combo (it could be anything)
                    static ImGuiComboFlags flags = 0;

                    if (ImGui::BeginCombo("Select Default Configuration", combo_preview_value, flags)) {
                        for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                            const bool is_selected = (presetItemIdIndex == n);
                            if (ImGui::Selectable(items[n], is_selected))
                                presetItemIdIndex = n;

                            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    btnConnect = ImGui::Button("connect", ImVec2(175.0f, 30.0f));
                    bool btnCancel = ImGui::Button("cancel", ImVec2(175.0f, 30.0f));
                    btnAutoConnect = ImGui::Button("Automatic configuration", ImVec2(175.0f, 30.0f));

                    if (btnCancel) {
                        ImGui::CloseCurrentPopup();
                    }

                    if (btnAutoConnect) {
                        // Attempt to search for camera
                        autoDetectCamera();
                    }

                    if (btnConnect) {
                        // Loop through devices and check that it doesn't exist already.
                        for (auto &d: devices) {
                            if (d.IP == inputIP && strcmp(items[presetItemIdIndex], "Virtual Camera") != 0)
                                deviceAlreadyExist = true;
                        }

                        if (!deviceAlreadyExist) {
                            createDefaultElement(inputName.data(), inputIP.data(), items[presetItemIdIndex]);
                            deviceAlreadyExist = true;
                        }

                    }


                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Advanced Options")) {
                    ImGui::Text("NOT IMPLEMENTED YET\n\nConnect to your MultiSense Device");
                    ImGui::Separator();
                    ImGui::InputText("Profile name##2", inputName.data(),
                                     inputFieldNameLength);
                    ImGui::InputText("Camera ip##1", inputIP.data(), inputFieldNameLength);
                    ImGui::Spacing();
                    ImGui::Spacing();

                    static int item = 1;
                    static float color[4] = {0.4f, 0.7f, 0.0f, 0.5f};
                    ImGui::Combo("Ethernet Adapter", &item, "lan\0eth0\0eth1\0eth2\0\0");

                    ImGui::Spacing();
                    ImGui::Spacing();

                    btnConnect = ImGui::Button("connect", ImVec2(175.0f, 30.0f));
                    ImGui::SameLine();

                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            // On connect button click
            if (btnConnect) {
                for (auto &d: devices) {
                    if (d.IP == inputIP)
                        deviceAlreadyExist = true;
                }

                if (!deviceAlreadyExist)
                    createAdvancedElement(inputName.data(), inputIP.data());

                ImGui::CloseCurrentPopup();

            }

            ImGui::EndPopup();

        }

        ImGui::Spacing();
        ImGui::Spacing();

        if (!devices.empty())
            sidebarElements();

        addDeviceButton();
        ImGui::End();
        ImGui::PopStyleColor(); // bg color

    }

private:

    std::vector<Element> devices;

    bool btnConnect = false;
    bool btnAdd = false;
    bool btnAutoConnect = false;

    static const uint32_t inputFieldNameLength = 32;
    uint32_t presetItemIdIndex = 0;
    std::string inputIP = "10.66.171.21";
    std::string inputName = "Profile 1";

    void createDefaultElement(char *name, char *ip, const char *cameraName) {
        Element el;

        el.name = name;
        el.IP = ip;
        el.state = ArJustAddedState;
        el.cameraName = cameraName;
        el.clicked = true;

        devices.emplace_back(el);

        handles->devices = &devices;

        Log::Logger::getInstance()->info("GUI:: Connect clicked for Default Device");
        Log::Logger::getInstance()->info("GUI:: Using: Ip: %s, and preset: %s", ip, name);


    }

    void createAdvancedElement(char *name, char *ip) {
        Element el;

        el.name = name;
        el.IP = ip;
        el.state = ArJustAddedState;
        el.cameraName = "Unknown";
        el.clicked = true;

        devices.emplace_back(el);

        handles->devices = &devices;

        Log::Logger::getInstance()->info("GUI:: Connect clicked for Advanced Device");
        Log::Logger::getInstance()->info("GUI:: Using: Ip: %s", ip);


    }

    void sidebarElements() {
        for (int i = 0; i < devices.size(); ++i) {
            auto &e = devices[i];

            std::string buttonIdentifier;
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
                    buttonIdentifier = "Disconnected";
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.713f, 0.065f, 0.066f, 1.0f));
                    break;
                case ArUnavailableState:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    buttonIdentifier = "Unavailable";
                    break;
                case ArJustAddedState:
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.03f, 0.07f, 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
                    buttonIdentifier = "Added...";
                    break;
            }

            ImGui::SetCursorPos(ImVec2(0, ImGui::GetCursorPosY()));

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            std::string winId = e.name + "Child";
            ImGui::BeginChild(winId.c_str(), ImVec2(handles->info->sidebarWidth, handles->info->elementHeight),
                              false, ImGuiWindowFlags_NoDecoration);

            ImGui::Dummy(ImVec2(0.0f, 20.0f));

            ImVec2 window_pos = ImGui::GetWindowPos();
            ImVec2 window_size = ImGui::GetWindowSize();
            ImVec2 window_center = ImVec2(window_pos.x + window_size.x * 0.5f, window_pos.y + window_size.y * 0.5f);
            ImVec2 cursorPos = ImGui::GetCursorPos();
            ImVec2 lineSize;
            // Profile Name
            {
                ImGui::PushFont(handles->info->font24);
                lineSize = ImGui::CalcTextSize(e.name.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(cursorPos);
                ImGui::Text("%s", e.name.c_str());
                ImGui::PopFont();
            }
            // Camera Name
            {
                ImGui::PushFont(handles->info->font13);
                lineSize = ImGui::CalcTextSize(e.cameraName.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
                ImGui::Text("%s", e.cameraName.c_str());
                ImGui::PopFont();
            }
            // Camera IP Address
            {
                ImGui::PushFont(handles->info->font13);
                lineSize = ImGui::CalcTextSize(e.IP.c_str());
                cursorPos.x = window_center.x - (lineSize.x / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));

                ImGui::Text("%s", e.IP.c_str());
                ImGui::PopFont();
            }

            // Status Button
            {
                ImGui::Dummy(ImVec2(0.0f, 5.0f));
                ImGui::PushFont(handles->info->font18);
                //ImGuiStyle style = ImGui::GetStyle();
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
                cursorPos.x = window_center.x - (ImGui::GetFontSize() * 10 / 2);
                ImGui::SetCursorPos(ImVec2(cursorPos.x, ImGui::GetCursorPosY()));
            }

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
