//
// Created by magnus on 4/19/22.
//

#ifndef MULTISENSE_SIDEBAR_H
#define MULTISENSE_SIDEBAR_H


#include <algorithm>
#include "imgui.h"
#include "Layer.h"
#include "imgui_internal.h"

class SideBar : public ArEngine::Layer {
public:
    void OnAttach() override {
    }

    void OnDetach() override {
    }

    void OnUIRender() override {
        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        float sidebarWidth = 250.0f;
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(sidebarWidth, info->height));
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
            bool btn = false;

            char *inputName = "Window view";
            char *inputIP = "10.42.0.10";

            ImGui::Text("Connect to your MultiSense Device");
            ImGui::Separator();
            ImGui::InputText("Profile name", inputName,
                             IM_ARRAYSIZE(inputName));
            ImGui::InputText("Camera ip", inputIP, IM_ARRAYSIZE(inputIP));

            btn = ImGui::Button("connect", ImVec2(175.0f, 30.0f));
            ImGui::SameLine();

            if (btn) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();

        }


        ImGui::SetCursorPos(ImVec2(20, 650));
        bool addBtn = false;
        addBtn = ImGui::Button("ADD DEVICE", ImVec2(200.0f, 35.0f));

        if (addBtn) {
            ImGui::OpenPopup("add_device_modal");

        }

        ImGui::End();

    }
};


#endif //MULTISENSE_SIDEBAR_H
