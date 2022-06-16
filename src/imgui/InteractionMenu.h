//
// Created by magnus on 5/5/22.
//

#ifndef MULTISENSE_INTERACTIONMENU_H
#define MULTISENSE_INTERACTIONMENU_H

#include "Layer.h"

class InteractionMenu : public Layer {
public:

// Create global object for convenience in other functions

    void onFinishedRender() override {

    }

    void OnUIRender(GuiObjectHandles *handles) override {

        bool pOpen = true;
        ImGuiWindowFlags window_flags = 0;
        window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
        ImGui::SetNextWindowPos(ImVec2(handles->info->sidebarWidth, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(handles->info->width - handles->info->sidebarWidth, handles->info->height / 4));
        ImGui::Begin("InteractionMenu", &pOpen, window_flags);

        if (handles->devices != nullptr) {
            for (auto &d: *handles->devices) {

                // Create dropdown
                if (d.state == ArActiveState) {
                    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_FittingPolicyResizeDown;
                    if (ImGui::BeginTabBar("InteractionTabs", tab_bar_flags)) {
                        if (ImGui::BeginTabItem("Streaming")) {
                            addStreamingTab(handles, d);

                            ImGui::EndTabItem();
                        }
                        if (ImGui::BeginTabItem("Configuration")) {
                            addConfigurationTab(handles, d);

                            ImGui::EndTabItem();
                        }
                        ImGui::EndTabBar();
                    }

                }
            }
        }

        ImGui::ShowDemoWindow();

        ImGui::End();
    }

    void addConfigurationTab(GuiObjectHandles *handles, Element d) {
        ImGui::Text("This tab is reserved for configurations. Placeholders, not implemented");

        ImGui::SliderFloat("Exposure time", &handles->sliderOne, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("LED duty cycle", &handles->sliderTwo, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);

    }

    void addStreamingTab(GuiObjectHandles *handles, Element d) {


        if (ImGui::BeginCombo("Resolution",
                              d.selectedStreamingMode.c_str())) // The second parameter is the label previewed before opening the combo.
        {
            for (auto &mode: d.modes) {

                bool is_selected = (mode.modeName ==
                                    d.selectedStreamingMode); // You can store your selection however you want, outside or inside your objects
                if (ImGui::Selectable(mode.modeName.c_str(), is_selected)) {
                    d.selectedStreamingMode = mode.modeName;

                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
            }
            ImGui::EndCombo();
        } // End dropdown

        ImGui::Checkbox("Depth Image", &d.depthImage);
        ImGui::Checkbox("Color Image", &d.colorImage);
        ImGui::Checkbox("Point Cloud", &d.pointCloud);

        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPos().x, ImGui::GetCursorPos().y + 10.0f));

        d.btnShowPreviewBar = ImGui::Button("Preview Bar", ImVec2(175.0f, 20.0f));

        ImGui::SliderFloat("X", &handles->sliderOne, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("Y", &handles->sliderTwo, -2.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
        ImGui::SliderFloat("Z", &handles->sliderThree, -5.0f, 2.0f, "%.3f", ImGuiSliderFlags_None);
    }


};

#endif //MULTISENSE_INTERACTIONMENU_H
